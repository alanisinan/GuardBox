"""Runtime / deployment benchmark (Reviewer 2 point 6; Reviewer 6).

Measures per-window inference latency for the LSTM autoencoder and the RF
classifier, plus model sizes, on this machine, to support a deployment-cost
paragraph. Writes results/runtime_<topo>.json.
"""

import argparse
import json
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from detection import config as C, train_lstm, train_rf, detector  # noqa: E402
from detection.windowing import scenario_windows  # noqa: E402
from experiments import datagen  # noqa: E402


def _cpu_descriptor():
    """A human-readable CPU + memory descriptor for the deployment section."""
    import subprocess

    def _sysctl(key):
        try:
            return subprocess.run(["sysctl", "-n", key], capture_output=True,
                                  text=True, timeout=5).stdout.strip()
        except Exception:
            return ""

    if sys.platform == "darwin":
        cpu = _sysctl("machdep.cpu.brand_string") or platform.machine()
        cores = _sysctl("hw.ncpu")
        membytes = _sysctl("hw.memsize")
        gb = f"{int(membytes) / 1e9:.0f} GB" if membytes.isdigit() else ""
        parts = [p for p in [cpu, f"{cores} cores" if cores else "", gb] if p]
        return ", ".join(parts)
    return platform.processor() or platform.machine()


def run(topo=C.DEFAULT_TOPOLOGY, n_eval=C.SCENARIOS_PER_CLASS, reps=200):
    C.ensure_dirs()
    splits = datagen.build_all(topo=topo, n_eval=n_eval)
    lstm_model, scaler, threshold, lstm_meta = train_lstm.train(
        splits["normal_train"], splits["normal_calib"], tag=f"{topo}_rt")
    rf_frame = train_rf.build_training_frame(splits["rf_train"])
    rf_clf, _ = train_rf.train(rf_frame, tag=f"{topo}_rt")

    df = splits["eval"][0]
    win, _ = scenario_windows(df, C.WINDOW_LENGTH, C.LSTM_FEATURES)
    one = train_lstm.scale_windows(win[:1], scaler)
    snap = df[C.FEATURES].to_numpy(dtype="float32")[:1]

    # warm-up
    lstm_model.predict(one, verbose=0)
    rf_clf.predict(snap)

    t = time.time()
    for _ in range(reps):
        lstm_model.predict(one, verbose=0)
    lstm_ms = (time.time() - t) / reps * 1e3

    t = time.time()
    for _ in range(reps):
        rf_clf.predict(snap)
    rf_ms = (time.time() - t) / reps * 1e3

    # Peak resident memory of the loaded-and-inferring detector (deployment footprint).
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_gb = peak / 1e9 if sys.platform == "darwin" else peak / 1e6

    lstm_path = os.path.join(C.MODELS_DIR, f"lstm_ae_{topo}_rt.keras")
    rf_path = os.path.join(C.MODELS_DIR, f"rf_{topo}_rt.pkl")
    results = {
        "machine": {
            "platform": platform.platform(),
            "processor": _cpu_descriptor(),
            "python": platform.python_version(),
        },
        "lstm_inference_ms_per_window": lstm_ms,
        "rf_inference_ms_per_snapshot": rf_ms,
        "combined_ms_per_decision": lstm_ms + rf_ms,
        "peak_inference_memory_gb": peak_gb,
        "lstm_params": lstm_meta["total_params"],
        "lstm_model_kb": os.path.getsize(lstm_path) / 1024 if os.path.exists(lstm_path) else None,
        "rf_model_kb": os.path.getsize(rf_path) / 1024 if os.path.exists(rf_path) else None,
        "snapshot_interval_s": C.SNAPSHOT_INTERVAL_S,
        "reps": reps,
    }
    out = os.path.join(C.RESULTS_DIR, f"runtime_{topo}.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[runtime] LSTM {lstm_ms:.2f} ms/window, RF {rf_ms:.3f} ms/snapshot, "
          f"combined {lstm_ms+rf_ms:.2f} ms << {C.SNAPSHOT_INTERVAL_S*1000:.0f} ms interval")
    print(f"[runtime] wrote {out}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--topo", default=C.DEFAULT_TOPOLOGY, choices=list(C.SUMO_CONFIGS))
    ap.add_argument("--n-eval", type=int, default=C.SCENARIOS_PER_CLASS)
    args = ap.parse_args()
    run(topo=args.topo, n_eval=args.n_eval)
