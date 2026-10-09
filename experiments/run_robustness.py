"""Robustness study (Reviewer 1 point 2; Reviewer 6 point 2).

Two experiments, both on the held-out live-evaluation set with the deployed
models (no retraining):

1. Sensor-noise sweep: additive Gaussian noise of increasing magnitude (a
   fraction of each feature's normal-operation standard deviation) is applied to
   the continuous telemetry features, and per-class detection rate + the normal
   false-positive rate are reported vs noise level.

2. Benign non-attack "hard negatives": scenarios that are NOT attacks but
   resemble one channel -- an authorized open-cabinet maintenance visit (door
   open, no vibration/temperature disturbance) and a brief emergency-preemption
   phase change -- are run through the detector to check that they are not
   misclassified as attacks.

Writes results/robustness.json.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from detection import config as C, train_lstm, train_rf, detector  # noqa: E402
from experiments import datagen  # noqa: E402
from experiments.run_evaluation import global_norm_stats  # noqa: E402

# continuous features to perturb with sensor noise (binary flags/derived left intact)
NOISE_FEATURES = [
    "elapsed_seconds", "phase_deviation_pct", "waiting_vehicles", "avg_speed",
    "co2_mg_s", "nox_mg_s", "traci_cmd_count", "traci_set_count_60s",
    "cmd_rate_delta", "vibration_level", "cabinet_temperature_c", "power_voltage",
]
NOISE_LEVELS = [0.0, 0.25, 0.5, 1.0]


def _feature_std(normal_dfs):
    stacked = pd.concat(normal_dfs, ignore_index=True)
    return {f: float(stacked[f].std() + 1e-9) for f in NOISE_FEATURES}


def _add_noise(df, stds, sigma, rng):
    out = df.copy()
    for f in NOISE_FEATURES:
        out[f] = out[f].to_numpy() + rng.normal(0.0, sigma * stds[f], size=len(out))
    return out


def noise_sweep(eval_dfs, stds, lstm, scaler, rf, thr, norm_stats):
    rows = {}
    for sigma in NOISE_LEVELS:
        rng = np.random.default_rng(int(sigma * 1000))
        counts = {c: {"n": 0, "det": 0} for c in C.CLASSES}
        for df in eval_dfs:
            ndf = df if sigma == 0 else _add_noise(df, stds, sigma, rng)
            rec = detector.detect_scenario(ndf, lstm, scaler, rf, thr,
                                           mode="full", norm_stats=norm_stats)
            counts[rec["label"]]["n"] += 1
            counts[rec["label"]]["det"] += int(rec["detected"])
        rows[str(sigma)] = {c: counts[c]["det"] / counts[c]["n"] if counts[c]["n"] else 0.0
                            for c in C.CLASSES}
    return rows


def _benign_maintenance(normal_dfs):
    """Authorized visit: door open across the mid window, no disturbance alarm,
    timing and commands normal. Should NOT be flagged."""
    out = []
    for df in normal_dfs:
        d = df.copy()
        m = (d["snapshot"] >= 30) & (d["snapshot"] < 90)
        d.loc[m, "door_state"] = 1
        # vibration/temperature stay normal, so sensor_anomaly_flag stays 0
        out.append(d)
    return out


def _preemption(normal_dfs, rng):
    """Brief emergency preemption: a short, moderate phase change (below the
    attack envelope), no extension flag, no command burst. Should NOT be an
    attack, though it tests the timing channel."""
    out = []
    for df in normal_dfs:
        d = df.copy()
        m = (d["snapshot"] >= 40) & (d["snapshot"] < 52)   # ~12 snapshots
        dev = rng.uniform(25.0, 45.0)
        d.loc[m, "phase_deviation_pct"] = dev
        d.loc[m, "elapsed_seconds"] = C.EXPECTED_PHASE_DURATION_S * (1 + dev / 100.0)
        out.append(d)
    return out


def hard_negatives(normal_eval, lstm, scaler, rf, thr, norm_stats):
    rng = np.random.default_rng(7)
    cases = {
        "benign_maintenance": _benign_maintenance(normal_eval),
        "emergency_preemption": _preemption(normal_eval, rng),
    }
    out = {}
    for name, dfs in cases.items():
        det = 0
        for d in dfs:
            # treat as a normal (no attack window) scenario: scan whole run for FP
            d2 = d.copy()
            d2["attack_active"] = 0
            rec = detector.detect_scenario(d2, lstm, scaler, rf, thr,
                                           mode="full", norm_stats=norm_stats)
            det += int(rec["detected"])
        out[name] = {"n": len(dfs), "flagged": det, "fp_rate": det / len(dfs) if dfs else 0.0}
    return out


def run(topo=C.DEFAULT_TOPOLOGY, n_eval=C.SCENARIOS_PER_CLASS, tag=None, out_json=None):
    C.ensure_dirs()
    tag = tag or topo
    splits = datagen.build_all(topo=topo, n_eval=n_eval, progress=False)
    lstm, scaler, thr, _ = train_lstm.load(tag=tag)
    rf, _ = train_rf.load(tag=tag)
    norm_stats, _ = global_norm_stats(splits["eval"], lstm, scaler, rf, C.WINDOW_LENGTH)
    stds = _feature_std(splits["normal_train"])

    sweep = noise_sweep(splits["eval"], stds, lstm, scaler, rf, thr, norm_stats)
    normal_eval = [df for df in splits["eval"] if df["label"].iloc[0] == "normal"]
    hardneg = hard_negatives(normal_eval, lstm, scaler, rf, thr, norm_stats)

    results = {"topology": topo, "noise_levels": NOISE_LEVELS,
               "noise_sweep": sweep, "hard_negatives": hardneg}
    out = out_json or os.path.join(C.RESULTS_DIR, "robustness.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[robustness] wrote {out}")
    print("noise sweep (detection rate; last col = normal FP):")
    for s, r in sweep.items():
        print(f"  sigma={s}: " + "  ".join(f"{c.split('_')[0]}={r[c]*100:.0f}%" for c in C.CLASSES))
    print("hard negatives (flagged / n; lower is better):")
    for name, hv in hardneg.items():
        print(f"  {name}: {hv['flagged']}/{hv['n']} = {hv['fp_rate']*100:.0f}%")
    return results


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--topo", default=C.DEFAULT_TOPOLOGY, choices=list(C.SUMO_CONFIGS))
    ap.add_argument("--n-eval", type=int, default=C.SCENARIOS_PER_CLASS)
    args = ap.parse_args()
    run(topo=args.topo, n_eval=args.n_eval)
