"""Rolling-window length study (Reviewer 6, point 6).

Re-runs detection with several window lengths (default 12 vs 18) and reports the
physical-intrusion detection-rate / latency trade-off, plus the effect on the
other classes and the false-positive rate. Writes results/window_study_<topo>.json.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from detection import config as C, train_lstm, train_rf, detector  # noqa: E402
from experiments import datagen  # noqa: E402
from experiments.run_evaluation import wilson_ci, global_norm_stats  # noqa: E402


def run(topo=C.DEFAULT_TOPOLOGY, windows=(12, 18), n_eval=C.SCENARIOS_PER_CLASS):
    C.ensure_dirs()
    splits = datagen.build_all(topo=topo, n_eval=n_eval)
    rf_frame = train_rf.build_training_frame(splits["rf_train"])
    rf_clf, _ = train_rf.train(rf_frame, tag=f"{topo}_ws")

    results = {"topology": topo, "windows": {}}
    for w in windows:
        lstm_model, scaler, threshold, _ = train_lstm.train(
            splits["normal_train"], splits["normal_calib"], window=w, tag=f"{topo}_w{w}")
        norm_stats, _ = global_norm_stats(splits["eval"], lstm_model, scaler, rf_clf, w)
        per_class = {c: {"n": 0, "det": 0, "lat": []} for c in C.CLASSES}
        for df in splits["eval"]:
            rec = detector.detect_scenario(df, lstm_model, scaler, rf_clf, threshold,
                                           window=w, mode="full", norm_stats=norm_stats)
            pc = per_class[rec["label"]]
            pc["n"] += 1
            pc["det"] += int(rec["detected"])
            if rec["detected"] and rec["first_detection_s"] is not None:
                pc["lat"].append(rec["first_detection_s"])
        summary = {}
        for c in C.CLASSES:
            pc = per_class[c]
            lo, hi = wilson_ci(pc["det"], pc["n"])
            summary[c] = {
                "detection_rate": pc["det"] / pc["n"] if pc["n"] else 0.0,
                "det": pc["det"], "n": pc["n"], "wilson_ci": [lo, hi],
                "mean_latency_s": float(np.mean(pc["lat"])) if pc["lat"] else None,
            }
        results["windows"][str(w)] = summary
        print(f"window={w}: physical={summary['physical_intrusion']['detection_rate']*100:.0f}% "
              f"lat={summary['physical_intrusion']['mean_latency_s']}  "
              f"normalFP={summary['normal']['detection_rate']*100:.0f}%")

    out = os.path.join(C.RESULTS_DIR, f"window_study_{topo}.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[window-study] wrote {out}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--topo", default=C.DEFAULT_TOPOLOGY, choices=list(C.SUMO_CONFIGS))
    ap.add_argument("--windows", type=int, nargs="+", default=[12, 18])
    ap.add_argument("--n-eval", type=int, default=C.SCENARIOS_PER_CLASS)
    args = ap.parse_args()
    run(topo=args.topo, windows=tuple(args.windows), n_eval=args.n_eval)
