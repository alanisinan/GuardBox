"""LSTM weight-initialization sensitivity of the main evaluation (Reviewer 3, round 2).

The published results/eval_<topo>.json was produced when train_lstm.py called
tf.random.set_seed only; under Keras 3 that does not fix the layer weight
initializers, so the reported LSTM was trained from an unseeded initialization.
This script retrains the autoencoder under several fixed seeds, reruns the full
main evaluation for each, and compares every quantity with the published file:

  * discrete outcomes (per-class detection counts and Wilson intervals, ablation,
    detection-latency bounds, RF attribution and cross-validation scores, and the
    confusion matrix) must match exactly;
  * continuous LSTM-derived quantities (per-class mean reconstruction error, the
    separation ratio, per-class mean severity) are reported as deviations.

The robustness study (noise sweep and hard negatives), which reuses the trained
models, is rerun for each seed as well, because its noisy-normal false-positive
rates depend on where the calibrated threshold falls.

Writes results/init_sensitivity.json (per-seed outputs go to
results/init_sensitivity/).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from detection import config as C  # noqa: E402
from experiments import run_evaluation, run_robustness  # noqa: E402

SEEDS = (42, 1, 2, 3)


def _discrete(ev):
    """Every discrete quantity of the main evaluation, in comparable form."""
    t1 = ev["table1_detection"]
    return {
        "detection": {c: [t1[c]["detected"], t1[c]["n"]] for c in C.CLASSES},
        "wilson_ci": {c: [round(x, 6) for x in t1[c]["wilson_ci"]] for c in C.CLASSES},
        "ablation": {m: {c: [r[c]["det"], r[c]["n"]] for c in C.CLASSES}
                     for m, r in ev["ablation"].items()},
        "latency_s": [ev["latency_s"]["min"], ev["latency_s"]["max"]],
        "rf_macro_f1": ev["rf_attribution_on_eval"]["macro_f1"],
        "rf_confusion_matrix": ev["rf_attribution_on_eval"]["confusion_matrix"],
        "rf_cv": [ev["rf_cv"]["f1_macro_mean"], ev["rf_cv"]["f1_macro_std"]],
    }


def run(topo=C.DEFAULT_TOPOLOGY, seeds=SEEDS):
    with open(os.path.join(C.RESULTS_DIR, f"eval_{topo}.json")) as fh:
        pub = json.load(fh)
    with open(os.path.join(C.RESULTS_DIR, "robustness.json")) as fh:
        pub_rob = json.load(fh)
    out_dir = os.path.join(C.RESULTS_DIR, "init_sensitivity")
    os.makedirs(out_dir, exist_ok=True)

    per_seed = {}
    for seed in seeds:
        print(f"\n### LSTM seed {seed}", flush=True)
        tag = f"{topo}_seed{seed}"
        ev = run_evaluation.run(topo=topo, tag=tag, lstm_seed=seed,
                                out_json=os.path.join(out_dir, f"eval_{tag}.json"))
        rob = run_robustness.run(topo=topo, tag=tag,
                                 out_json=os.path.join(out_dir, f"robustness_{tag}.json"))
        recon_dev = {
            c: (ev["table1_detection"][c]["mean_recon_error"]
                / pub["table1_detection"][c]["mean_recon_error"] - 1.0)
            for c in C.CLASSES}
        sev_diff = {
            c: abs(ev["table1_detection"][c]["mean_severity"]
                   - pub["table1_detection"][c]["mean_severity"])
            for c in C.CLASSES}
        per_seed[str(seed)] = {
            "discrete_identical_to_published": _discrete(ev) == _discrete(pub),
            "lstm_threshold": ev["lstm_meta"]["threshold"],
            "recon_mean_rel_dev": recon_dev,
            "severity_abs_diff": sev_diff,
            "separation_ratio": ev["separation"]["separation_ratio"],
            "noise_sweep": rob["noise_sweep"],
            "hard_negatives": {k: v["flagged"] for k, v in rob["hard_negatives"].items()},
        }

    summary = {
        "topology": topo,
        "seeds": list(seeds),
        "published_separation_ratio": pub["separation"]["separation_ratio"],
        "all_discrete_identical": all(s["discrete_identical_to_published"]
                                      for s in per_seed.values()),
        "max_abs_recon_rel_dev": max(abs(d) for s in per_seed.values()
                                     for d in s["recon_mean_rel_dev"].values()),
        "max_severity_abs_diff": max(d for s in per_seed.values()
                                     for d in s["severity_abs_diff"].values()),
        "separation_ratio_range": [min(s["separation_ratio"] for s in per_seed.values()),
                                   max(s["separation_ratio"] for s in per_seed.values())],
        "noise_sweep_range": {
            sigma: {c: [min(s["noise_sweep"][sigma][c] for s in per_seed.values()),
                        max(s["noise_sweep"][sigma][c] for s in per_seed.values())]
                    for c in C.CLASSES}
            for sigma in pub_rob["noise_sweep"]},
        "published_noise_sweep": pub_rob["noise_sweep"],
        "hard_negatives_identical": all(
            s["hard_negatives"] == {k: v["flagged"] for k, v in pub_rob["hard_negatives"].items()}
            for s in per_seed.values()),
        "per_seed": per_seed,
    }
    out = os.path.join(C.RESULTS_DIR, "init_sensitivity.json")
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n[init-sensitivity] discrete outcomes identical for all seeds: "
          f"{summary['all_discrete_identical']}; max |recon dev| = "
          f"{summary['max_abs_recon_rel_dev']*100:.1f}%; separation ratio "
          f"{summary['separation_ratio_range'][0]:.0f}-{summary['separation_ratio_range'][1]:.0f}; "
          f"max |severity diff| = {summary['max_severity_abs_diff']:.4f}")
    print(f"[init-sensitivity] wrote {out}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--topo", default=C.DEFAULT_TOPOLOGY, choices=list(C.SUMO_CONFIGS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = ap.parse_args()
    run(topo=args.topo, seeds=tuple(args.seeds))
