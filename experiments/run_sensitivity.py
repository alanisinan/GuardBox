"""Feature-correlation and severity-weight sensitivity analysis (Reviewer 2.5).

Produces:
  * a Pearson feature-correlation matrix over the live-evaluation snapshots
    (figure fig7_feature_correlation.png), used to discuss feature redundancy
    and selection;
  * a severity-weight sensitivity study that recomputes per-class mean severity
    under several weight vectors and reports whether the class ordering is
    stable, so the score's robustness to the exact weights is quantified rather
    than asserted.

Writes results/sensitivity.json.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from detection import config as C, train_lstm, train_rf, detector  # noqa: E402
from experiments import datagen  # noqa: E402
from experiments.run_evaluation import global_norm_stats  # noqa: E402

OVERLEAF = "/Users/sinan/Downloads/GuardBox_Overleaf"

WEIGHT_VECTORS = {
    "default (0.4/0.3/0.2/0.1)": (0.4, 0.3, 0.2, 0.1),
    "equal (0.25 each)": (0.25, 0.25, 0.25, 0.25),
    "recon-heavy (0.6/0.2/0.1/0.1)": (0.6, 0.2, 0.1, 0.1),
    "phase-heavy (0.2/0.5/0.2/0.1)": (0.2, 0.5, 0.2, 0.1),
    "sensor-heavy (0.2/0.2/0.5/0.1)": (0.2, 0.2, 0.5, 0.1),
}


def _mm(x, lo, hi):
    if hi <= lo:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def run(topo=C.DEFAULT_TOPOLOGY, n_eval=C.SCENARIOS_PER_CLASS):
    C.ensure_dirs()
    splits = datagen.build_all(topo=topo, n_eval=n_eval, progress=False)
    lstm_model, scaler, threshold, _ = train_lstm.load(tag=topo)
    rf_clf, _ = train_rf.load(tag=topo)
    norm_stats, _ = global_norm_stats(splits["eval"], lstm_model, scaler, rf_clf, C.WINDOW_LENGTH)

    # --- Severity components per scenario ---
    comp = {c: [] for c in C.CLASSES}   # (e, p, sensor, rf_conf)
    for df in splits["eval"]:
        sig = detector.scenario_signals(df, lstm_model, scaler, rf_clf, C.WINDOW_LENGTH)
        e = _mm(np.array([np.max(sig["recon"])]), *norm_stats["recon"])[0]
        p = _mm(np.array([np.max(sig["phase_dev"])]), *norm_stats["phase_dev"])[0]
        s = float(np.max(sig["sensor_persist"]))
        rc = float(np.max(sig["rf_conf"]))
        comp[df["label"].iloc[0]].append((e, p, s, rc))

    order = ["timing_manipulation", "replay_attack", "physical_intrusion",
             "combined_attack", "normal"]
    sensitivity = {}
    ref_rank = None
    for name, w in WEIGHT_VECTORS.items():
        means = {}
        for c in C.CLASSES:
            arr = np.array(comp[c])
            sev = w[0] * arr[:, 0] + w[1] * arr[:, 1] + w[2] * arr[:, 2] + w[3] * arr[:, 3]
            means[c] = float(np.mean(sev))
        ranked = sorted(C.CLASSES, key=lambda c: means[c], reverse=True)
        sensitivity[name] = {"means": means, "ranking": ranked}
        if ref_rank is None:
            ref_rank = ranked

    # Rank stability vs the default ordering (Spearman over class means).
    ref_vals = [sensitivity["default (0.4/0.3/0.2/0.1)"]["means"][c] for c in C.CLASSES]
    for name in sensitivity:
        vals = [sensitivity[name]["means"][c] for c in C.CLASSES]
        rho, _ = spearmanr(ref_vals, vals)
        sensitivity[name]["spearman_vs_default"] = float(rho)

    # --- Feature correlation matrix over eval snapshots ---
    all_rows = pd.concat(splits["eval"], ignore_index=True)
    corr = all_rows[C.FEATURES].corr(method="pearson").fillna(0.0)
    _plot_correlation(corr)

    # Highly correlated feature pairs (|r| >= 0.8), for the selection discussion.
    pairs = []
    feats = list(C.FEATURES)
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            r = float(corr.iloc[i, j])
            if abs(r) >= 0.8:
                pairs.append((feats[i], feats[j], round(r, 3)))

    results = {
        "topology": topo,
        "severity_sensitivity": sensitivity,
        "class_order_stable": all(sensitivity[n]["ranking"] == ref_rank for n in sensitivity),
        "high_corr_pairs_ge_0.8": pairs,
    }
    out = os.path.join(C.RESULTS_DIR, "sensitivity.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[sensitivity] wrote {out}")
    print(f"  class-order stable across all weight vectors: {results['class_order_stable']}")
    for name, d in sensitivity.items():
        print(f"  {name:32s} rho={d['spearman_vs_default']:.3f} "
              f"top={d['ranking'][0].split('_')[0]}")
    print(f"  highly-correlated pairs (|r|>=0.8): {len(pairs)}")
    _emit_table(sensitivity, order)
    return results


def _plot_correlation(corr):
    labels = [f.replace("_", " ") for f in corr.columns]
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson correlation")
    ax.set_title("Feature correlation matrix (live-evaluation snapshots)")
    ax.grid(False)
    for d in (C.FIG_DIR, OVERLEAF):
        fig.savefig(os.path.join(d, "fig7_feature_correlation.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig7_feature_correlation.png")


def _emit_table(sensitivity, order):
    short = {"timing_manipulation": "timing", "replay_attack": "replay",
             "physical_intrusion": "physical", "combined_attack": "combined",
             "normal": "normal"}
    rows = []
    for name, d in sensitivity.items():
        cells = " & ".join(f"{d['means'][c]:.3f}" for c in order)
        rows.append(f"{name} & {cells} \\\\")
    header = " & ".join(short[c] for c in order)
    body = ("\\begin{table}[!htbp]\n\\caption{Severity-weight sensitivity: per-class mean "
            "severity under alternative weight vectors $(w_1,w_2,w_3,w_4)$. Normal is the "
            "lowest-severity class and combined attack is the highest under every vector except "
            "reconstruction-heavy weighting, where replay's large reconstruction error makes it "
            "the highest; the middle ordering shifts with the weights, which is consistent with "
            "severity being an operator-tunable triage aid rather than an absolute ranking.}\n"
            "\\label{tab:sensitivity}\n\\centering\n\\scriptsize\n"
            "\\begin{tabular}{lccccc}\n\\toprule\nWeights $(w_1,w_2,w_3,w_4)$ & " + header
            + " \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    with open(os.path.join(OVERLEAF, "gb_tab_sensitivity.tex"), "w") as fh:
        fh.write(body)
    print("  wrote gb_tab_sensitivity.tex")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--topo", default=C.DEFAULT_TOPOLOGY, choices=list(C.SUMO_CONFIGS))
    ap.add_argument("--n-eval", type=int, default=C.SCENARIOS_PER_CLASS)
    args = ap.parse_args()
    run(topo=args.topo, n_eval=args.n_eval)
