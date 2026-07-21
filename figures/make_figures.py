"""Regenerate manuscript Figures 2-6 from the evaluation results.

Every figure is a clean, final presentation with no debugging annotations
(addresses Reviewer 5, Reviewer 6 point 1, and Reviewer 4). Output PNGs use the
exact filenames referenced by GuardBox_Overleaf/main.tex so they drop straight
into the manuscript.

Figures:
  2  phase deviation over simulation time (normal vs timing attack)
  3  RF confusion matrix on the 375-scenario evaluation set
  4  LSTM reconstruction-error distribution by class (log10 mg-free)
  5  severity score distribution by class
  6  top-10 RF feature importances
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detection import config as C  # noqa: E402

OVERLEAF = "/Users/sinan/Downloads/GuardBox_Overleaf"
PRIMARY = C.DEFAULT_TOPOLOGY
CLASS_LABELS = {
    "normal": "normal", "timing_manipulation": "timing",
    "replay_attack": "replay", "physical_intrusion": "physical",
    "combined_attack": "combined",
}
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def _load(name):
    with open(os.path.join(C.RESULTS_DIR, name)) as fh:
        return json.load(fh)


def _save(fig, fname):
    for d in (C.FIG_DIR, OVERLEAF):
        fig.savefig(os.path.join(d, fname), dpi=150)
    plt.close(fig)
    print(f"  wrote {fname}")


def fig2_phase_deviation():
    """Phase deviation timeline: representative normal vs timing scenario."""
    import pandas as pd
    sd = os.path.join(C.DATA_DIR, "scenarios", PRIMARY, "eval")
    normal = pd.read_csv(os.path.join(sd, "normal_5000.csv"))
    timing = pd.read_csv(os.path.join(sd, "timing_manipulation_6000.csv"))

    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(normal.sim_time_s, normal.phase_deviation_pct, label="normal scenario",
            color="#2c7fb8", lw=1.4)
    ax.plot(timing.sim_time_s, timing.phase_deviation_pct, label="timing manipulation",
            color="#d95f02", lw=1.4)
    onset = (C.WARMUP_SNAPSHOTS + C.ATTACK_START_SNAPSHOT) * C.SNAPSHOT_INTERVAL_S
    end = onset + C.ATTACK_DURATION_SNAPSHOTS * C.SNAPSHOT_INTERVAL_S
    ax.axvspan(onset, end, color="#d95f02", alpha=0.12, label="attack window")
    ax.set_xlabel("Simulation time (s)")
    ax.set_ylabel("Phase deviation (%)")
    ax.set_title("Phase deviation over time: normal vs timing manipulation")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95,
              handlelength=1.4, handletextpad=0.5, labelspacing=0.3, borderpad=0.4)
    _save(fig, "fig2_phase_deviation.png")


def fig3_confusion(res):
    cm = np.array(res["rf_attribution_on_eval"]["confusion_matrix"])
    labels = [CLASS_LABELS[c] for c in res["rf_attribution_on_eval"]["labels"]]
    fig, ax = plt.subplots(figsize=(6, 5.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=11)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("RF attribution confusion matrix (375-scenario evaluation)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="scenario count")
    _save(fig, "fig3_confusion_matrix.png")


def fig4_reconstruction(res):
    """Per-class reconstruction-error distribution from cached scenario recon."""
    import pandas as pd
    from detection import train_lstm, train_rf, detector
    from experiments import datagen
    splits = datagen.build_all(topo=PRIMARY, n_eval=C.SCENARIOS_PER_CLASS, progress=False)
    lstm_model, scaler, threshold, _ = train_lstm.load(tag=PRIMARY)
    rf_clf, _ = train_rf.load(tag=PRIMARY)

    by_class = {c: [] for c in C.CLASSES}
    for df in splits["eval"]:
        sig = detector.scenario_signals(df, lstm_model, scaler, rf_clf, C.WINDOW_LENGTH)
        by_class[df.label.iloc[0]].append(float(np.max(sig["recon"])))

    fig, ax = plt.subplots(figsize=(7.5, 4))
    data = [np.log10(np.array(by_class[c]) + 1e-4) for c in C.CLASSES]
    parts = ax.violinplot(data, showmeans=True, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_alpha(0.6)
    ax.set_xticks(range(1, len(C.CLASSES) + 1),
                  [CLASS_LABELS[c] for c in C.CLASSES])
    ax.set_ylabel(r"$\log_{10}$(reconstruction error MSE $+\,10^{-4}$)")
    ax.set_xlabel("Scenario class")
    ax.set_title("LSTM reconstruction-error distribution by class")
    _save(fig, "fig4_reconstruction_error_by_class.png")


def fig5_severity(res):
    import pandas as pd
    from detection import train_lstm, train_rf, detector
    from experiments import datagen
    from experiments.run_evaluation import global_norm_stats
    splits = datagen.build_all(topo=PRIMARY, n_eval=C.SCENARIOS_PER_CLASS, progress=False)
    lstm_model, scaler, threshold, _ = train_lstm.load(tag=PRIMARY)
    rf_clf, _ = train_rf.load(tag=PRIMARY)
    norm_stats, _ = global_norm_stats(splits["eval"], lstm_model, scaler, rf_clf, C.WINDOW_LENGTH)

    by_class = {c: [] for c in C.CLASSES}
    for df in splits["eval"]:
        rec = detector.detect_scenario(df, lstm_model, scaler, rf_clf, threshold,
                                       mode="full", norm_stats=norm_stats)
        by_class[rec["label"]].append(rec["severity"])

    fig, ax = plt.subplots(figsize=(7.5, 4))
    order = ["timing_manipulation", "replay_attack", "physical_intrusion",
             "combined_attack", "normal"]
    data = [by_class[c] for c in order]
    colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6)
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.55)
    for i, c in enumerate(order, 1):
        y = data[i - 1]
        x = np.random.default_rng(i).normal(i, 0.05, size=len(y))
        ax.scatter(x, y, s=8, color="#333333", alpha=0.4, zorder=3)
    ax.set_xticks(range(1, len(order) + 1), [CLASS_LABELS[c] for c in order])
    ax.set_ylabel("Severity score")
    ax.set_xlabel("Scenario class")
    ax.set_title("Severity score distribution by class")
    _save(fig, "fig5_severity_score.png")


def fig6_importances(res):
    imp = res["rf_feature_importances"][:10][::-1]
    names = [f for f, _ in imp]
    vals = [v for _, v in imp]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.barh(range(len(names)), vals, color="#2c7fb8", alpha=0.85)
    ax.set_yticks(range(len(names)), names)
    for i, v in enumerate(vals):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("Gini importance")
    ax.set_title("Top 10 Random Forest feature importances")
    ax.grid(axis="y", alpha=0)
    _save(fig, "fig6_feature_importances.png")


def main():
    res = _load(f"eval_{PRIMARY}.json")
    print("[figures] regenerating clean figures...")
    fig2_phase_deviation()
    fig3_confusion(res)
    fig4_reconstruction(res)
    fig5_severity(res)
    fig6_importances(res)
    print("[figures] done")


if __name__ == "__main__":
    main()
