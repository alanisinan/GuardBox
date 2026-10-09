"""Main GuardBox evaluation driver.

Generates (or loads) all scenario splits, trains the LSTM autoencoder and the RF
classifier, runs the detector over the 375-scenario live-evaluation set, and
writes every headline number the manuscript reports to results/eval_<topo>.json:

  * Table 1  : per-class detection rate, mean severity, mean reconstruction error
  * latency  : first-detection time range (simulation seconds)
  * separation ratio (with an explicit, reproducible derivation)
  * Table 2  : per-class RF precision / recall / F1 on the SAME 375-scenario set
  * Table 3  : ablation (lstm-only / rf-only / sensor-only / GuardBox)
  * RF cross-validation F1, feature importances, confusion matrix
  * Wilson 95% confidence intervals for every detection rate
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from detection import config as C  # noqa: E402
from detection import train_lstm, train_rf, detector  # noqa: E402
from experiments import datagen  # noqa: E402


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def global_norm_stats(eval_dfs, lstm_model, scaler, rf_clf, window):
    """Global (min,max) of reconstruction error and phase deviation over the
    evaluation set, for severity min-max normalization (Eq. 2)."""
    recon_all, phase_all = [], []
    sigs = []
    for df in eval_dfs:
        sig = detector.scenario_signals(df, lstm_model, scaler, rf_clf, window)
        sigs.append((df, sig))
        if sig is not None:
            recon_all.append(sig["recon"])
            phase_all.append(sig["phase_dev"])
    recon_all = np.concatenate(recon_all)
    phase_all = np.concatenate(phase_all)
    stats = {
        "recon": (float(recon_all.min()), float(recon_all.max())),
        "phase_dev": (float(phase_all.min()), float(phase_all.max())),
    }
    return stats, sigs


def run(topo=C.DEFAULT_TOPOLOGY, n_eval=C.SCENARIOS_PER_CLASS, n_rf_train=10,
        n_normal_train=60, n_normal_calib=20, window=C.WINDOW_LENGTH,
        tag=None, out_json=None, lstm_seed=C.RANDOM_SEED):
    C.ensure_dirs()
    tag = tag or topo
    t0 = time.time()
    splits = datagen.build_all(topo=topo, n_eval=n_eval, n_rf_train=n_rf_train,
                               n_normal_train=n_normal_train,
                               n_normal_calib=n_normal_calib)
    print(f"[eval] data ready in {time.time()-t0:.0f}s", flush=True)

    # --- Train LSTM autoencoder (normal-only, strict split) ---
    t1 = time.time()
    lstm_model, scaler, threshold, lstm_meta = train_lstm.train(
        splits["normal_train"], splits["normal_calib"], window=window, tag=tag,
        seed=lstm_seed)
    print(f"[eval] LSTM trained in {time.time()-t1:.0f}s, threshold={threshold:.4g}", flush=True)

    # --- Train RF (rf_train split) ---
    rf_frame = train_rf.build_training_frame(splits["rf_train"])
    rf_clf, rf_meta = train_rf.train(rf_frame, tag=tag)
    print(f"[eval] RF trained: CV F1={rf_meta['cv_f1_macro_mean']:.4f} "
          f"+/- {rf_meta['cv_f1_macro_std']:.4f}", flush=True)

    eval_dfs = splits["eval"]
    norm_stats, sigs = global_norm_stats(eval_dfs, lstm_model, scaler, rf_clf, window)

    # --- Per-scenario detection (full GuardBox) ---
    per_class = {c: {"n": 0, "detected": 0, "sev": [], "recon": [],
                     "latency": []} for c in C.CLASSES}
    rf_true, rf_pred_attack = [], []
    for df in eval_dfs:
        rec = detector.detect_scenario(df, lstm_model, scaler, rf_clf, threshold,
                                       window=window, mode="full", norm_stats=norm_stats)
        cls = rec["label"]
        pc = per_class[cls]
        pc["n"] += 1
        pc["detected"] += int(rec["detected"])
        pc["sev"].append(rec["severity"])
        pc["recon"].append(rec["mean_recon_error"])
        if rec["detected"] and rec["first_detection_s"] is not None:
            pc["latency"].append(rec["first_detection_s"])
        rf_true.append(cls)
        rf_pred_attack.append(rec["rf_pred_attack_window"])

    table1 = {}
    all_latencies = []
    for c in C.CLASSES:
        pc = per_class[c]
        rate = pc["detected"] / pc["n"] if pc["n"] else 0.0
        lo, hi = wilson_ci(pc["detected"], pc["n"])
        table1[c] = {
            "n": pc["n"],
            "detected": pc["detected"],
            "detection_rate": rate,
            "wilson_ci": [lo, hi],
            "mean_severity": float(np.mean(pc["sev"])) if pc["sev"] else 0.0,
            "mean_recon_error": float(np.mean(pc["recon"])) if pc["recon"] else 0.0,
        }
        if c != "normal":
            all_latencies.extend(pc["latency"])
    # normal reported as FP rate
    table1["normal"]["fp_rate"] = table1["normal"]["detection_rate"]

    # --- Separation ratio with explicit derivation ---
    normal_recon = table1["normal"]["mean_recon_error"]
    attack_recons = [table1[c]["mean_recon_error"] for c in C.CLASSES if c != "normal"]
    # aggregate attack recon = mean over all attack SCENARIOS (window-weighted equivalent)
    attack_scen_recon = []
    for c in C.CLASSES:
        if c != "normal":
            attack_scen_recon.extend(per_class[c]["recon"])
    agg_attack_recon = float(np.mean(attack_scen_recon))
    separation = {
        "normal_mean_recon": normal_recon,
        "attack_mean_recon_over_all_attack_scenarios": agg_attack_recon,
        "separation_ratio": agg_attack_recon / normal_recon if normal_recon else None,
        "per_class_ratio": {c: (table1[c]["mean_recon_error"] / normal_recon)
                            for c in C.CLASSES if c != "normal"},
        "derivation": "attack_mean_recon is the mean per-scenario mean reconstruction "
                      "error over all 300 attack scenarios (75 x 4 families); "
                      "separation_ratio = attack_mean_recon / normal_mean_recon.",
    }

    # --- RF attribution on the SAME 375-scenario set ---
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    rf_report = classification_report(rf_true, rf_pred_attack, output_dict=True,
                                      zero_division=0, labels=C.CLASSES)
    rf_cm = confusion_matrix(rf_true, rf_pred_attack, labels=C.CLASSES).tolist()
    rf_macro_f1 = float(f1_score(rf_true, rf_pred_attack, average="macro", labels=C.CLASSES))

    # --- Ablation ---
    ablation = {}
    for mode in ["lstm-only", "rf-only", "sensor-only", "full"]:
        counts = {c: {"n": 0, "det": 0} for c in C.CLASSES}
        for df in eval_dfs:
            rec = detector.detect_scenario(df, lstm_model, scaler, rf_clf, threshold,
                                           window=window, mode=mode, norm_stats=norm_stats)
            counts[rec["label"]]["n"] += 1
            counts[rec["label"]]["det"] += int(rec["detected"])
        ablation[mode] = {c: {"rate": counts[c]["det"] / counts[c]["n"] if counts[c]["n"] else 0.0,
                              "det": counts[c]["det"], "n": counts[c]["n"]}
                          for c in C.CLASSES}

    results = {
        "topology": topo,
        "window_length": window,
        "n_eval_per_class": n_eval,
        "table1_detection": table1,
        "latency_s": {
            "min": float(np.min(all_latencies)) if all_latencies else None,
            "max": float(np.max(all_latencies)) if all_latencies else None,
            "mean": float(np.mean(all_latencies)) if all_latencies else None,
        },
        "separation": separation,
        "rf_attribution_on_eval": {
            "report": rf_report, "confusion_matrix": rf_cm,
            "labels": C.CLASSES, "macro_f1": rf_macro_f1,
        },
        "rf_cv": {"f1_macro_mean": rf_meta["cv_f1_macro_mean"],
                  "f1_macro_std": rf_meta["cv_f1_macro_std"]},
        "rf_feature_importances": rf_meta["feature_importances"],
        "lstm_meta": lstm_meta,
        "ablation": ablation,
        "severity_norm_stats": norm_stats,
        "runtime_s": time.time() - t0,
    }

    out_json = out_json or os.path.join(C.RESULTS_DIR, f"eval_{tag}.json")
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[eval] wrote {out_json} in {results['runtime_s']:.0f}s total", flush=True)
    _print_summary(results)
    return results


def _print_summary(r):
    print("\n=== Table 1: detection by class ===")
    for c in C.CLASSES:
        t = r["table1_detection"][c]
        print(f"  {c:22s} det={t['detected']}/{t['n']} "
              f"rate={t['detection_rate']*100:5.1f}%  sev={t['mean_severity']:.3f}  "
              f"recon={t['mean_recon_error']:.4f}")
    print(f"latency {r['latency_s']['min']}-{r['latency_s']['max']} s")
    print(f"separation ratio ~ {r['separation']['separation_ratio']:.0f}x")
    print(f"RF macro-F1 on 375-eval = {r['rf_attribution_on_eval']['macro_f1']:.4f}; "
          f"CV F1 = {r['rf_cv']['f1_macro_mean']:.4f}")
    print("=== ablation (detection rate) ===")
    for mode in ["lstm-only", "rf-only", "sensor-only", "full"]:
        row = r["ablation"][mode]
        print(f"  {mode:12s} " + "  ".join(
            f"{c.split('_')[0]}={row[c]['rate']*100:.0f}%" for c in C.CLASSES))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--topo", default=C.DEFAULT_TOPOLOGY, choices=list(C.SUMO_CONFIGS))
    ap.add_argument("--n-eval", type=int, default=C.SCENARIOS_PER_CLASS)
    ap.add_argument("--n-rf-train", type=int, default=10)
    ap.add_argument("--n-normal-train", type=int, default=60)
    ap.add_argument("--n-normal-calib", type=int, default=20)
    ap.add_argument("--window", type=int, default=C.WINDOW_LENGTH)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--lstm-seed", type=int, default=C.RANDOM_SEED)
    args = ap.parse_args()
    run(topo=args.topo, n_eval=args.n_eval, n_rf_train=args.n_rf_train,
        n_normal_train=args.n_normal_train, n_normal_calib=args.n_normal_calib,
        window=args.window, tag=args.tag, lstm_seed=args.lstm_seed)
