"""Multi-intersection generalization study (Reviewers 1, 2, 6).

Retrains and evaluates the full GuardBox pipeline independently on each SUMO
topology (2x2 simple, 3x3 arterial, 3x3 demand variant) with per-topology
threshold calibration, then reports how detection rates and the top RF feature
importances shift across configurations. Writes results/multi_intersection.json.
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


def eval_topology(topo, n_eval):
    splits = datagen.build_all(topo=topo, n_eval=n_eval)
    lstm_model, scaler, threshold, _ = train_lstm.train(
        splits["normal_train"], splits["normal_calib"], tag=f"{topo}_mi")
    rf_frame = train_rf.build_training_frame(splits["rf_train"])
    rf_clf, rf_meta = train_rf.train(rf_frame, tag=f"{topo}_mi")
    norm_stats, _ = global_norm_stats(splits["eval"], lstm_model, scaler, rf_clf, C.WINDOW_LENGTH)

    per_class = {c: {"n": 0, "det": 0} for c in C.CLASSES}
    for df in splits["eval"]:
        rec = detector.detect_scenario(df, lstm_model, scaler, rf_clf, threshold,
                                       mode="full", norm_stats=norm_stats)
        per_class[rec["label"]]["n"] += 1
        per_class[rec["label"]]["det"] += int(rec["detected"])
    detection = {}
    for c in C.CLASSES:
        pc = per_class[c]
        lo, hi = wilson_ci(pc["det"], pc["n"])
        detection[c] = {"rate": pc["det"] / pc["n"] if pc["n"] else 0.0,
                        "det": pc["det"], "n": pc["n"], "wilson_ci": [lo, hi]}
    return {
        "detection": detection,
        "rf_cv_f1": rf_meta["cv_f1_macro_mean"],
        "top_features": rf_meta["feature_importances"][:6],
        "lstm_threshold": threshold,
    }


def run(topos=("intersection", "better_intersection", "complex_intersection"),
        n_eval=25):
    C.ensure_dirs()
    results = {"n_eval_per_class": n_eval, "topologies": {}}
    for topo in topos:
        print(f"\n### topology {topo}")
        results["topologies"][topo] = eval_topology(topo, n_eval)
        d = results["topologies"][topo]["detection"]
        print("  " + "  ".join(f"{c.split('_')[0]}={d[c]['rate']*100:.0f}%" for c in C.CLASSES))
    out = os.path.join(C.RESULTS_DIR, "multi_intersection.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[multi-intersection] wrote {out}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--topos", nargs="+",
                    default=["intersection", "better_intersection", "complex_intersection"])
    ap.add_argument("--n-eval", type=int, default=25)
    args = ap.parse_args()
    run(topos=tuple(args.topos), n_eval=args.n_eval)
