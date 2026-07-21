"""GuardBox detector and fusion logic.

Consumes a scenario's telemetry, slides a rolling window, computes LSTM
reconstruction error, applies the RF classifier, and fuses the signals into an
anomaly decision, an attack-class prediction, and a severity score.

Detection channels (single, unambiguous definitions)
----------------------------------------------------
1. Temporal channel  : LSTM reconstruction error above the calibrated threshold.
2. Physical channel  : sensor-anomaly flag asserted in at least
                       ``SENSOR_PERSISTENCE_FRACTION`` of the window's snapshots.

Detector modes
--------------
full (GuardBox) : window anomalous iff (temporal channel) OR (physical channel).
lstm-only       : window anomalous iff (temporal channel).            [temporal ablation]
rf-only         : window anomalous iff RF predicts a non-normal class. [attribution-as-detector ablation]
sensor-only     : window anomalous iff (physical channel).            [physical ablation]

A scenario is "detected" if any evaluated window is anomalous. The reported
attack class is the RF prediction on the most anomalous window's last snapshot.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection import config as C  # noqa: E402
from detection.windowing import scenario_windows  # noqa: E402
from detection.train_lstm import scale_windows, window_recon_errors  # noqa: E402


def scenario_signals(df, lstm_model, scaler, rf_clf, window=C.WINDOW_LENGTH):
    """Compute per-window signals for one scenario.

    Returns a dict with arrays over evaluated windows.
    """
    lstm_windows, last_idx = scenario_windows(df, window, C.LSTM_FEATURES)
    if len(lstm_windows) == 0:
        return None
    scaled = scale_windows(lstm_windows, scaler)
    recon = window_recon_errors(lstm_model, scaled)

    # Physical channel: accumulated flagged-snapshot count within each window.
    flags = df["sensor_anomaly_flag"].to_numpy()
    n = len(df)
    sensor_count = np.array([
        int(flags[end - window:end].sum()) for end in range(window, n + 1)
    ])
    sensor_persist = sensor_count >= C.SENSOR_PERSIST_COUNT

    # RF prediction + confidence on each window's last snapshot.
    last_snaps = df[C.FEATURES].to_numpy(dtype="float32")[last_idx]
    rf_pred = rf_clf.predict(last_snaps)
    rf_proba = rf_clf.predict_proba(last_snaps)
    rf_conf = rf_proba.max(axis=1)

    return {
        "recon": recon,
        "sensor_persist": sensor_persist,
        "rf_pred": rf_pred,
        "rf_conf": rf_conf,
        "last_idx": last_idx,
        "phase_dev": df["phase_deviation_pct"].to_numpy()[last_idx],
        "sensor_flag": flags[last_idx],
    }


def window_anomaly(sig, threshold, mode="full"):
    """Boolean per-window anomaly decision for the given detector mode."""
    temporal = sig["recon"] > threshold
    physical = sig["sensor_persist"]
    rf_anom = sig["rf_pred"] != "normal"
    if mode == "full":
        return temporal | physical
    if mode == "lstm-only":
        return temporal
    if mode == "rf-only":
        return rf_anom
    if mode == "sensor-only":
        return physical
    raise ValueError(f"unknown mode {mode}")


def detect_scenario(df, lstm_model, scaler, rf_clf, threshold,
                    window=C.WINDOW_LENGTH, mode="full", norm_stats=None):
    """Return a per-scenario detection record."""
    sig = scenario_signals(df, lstm_model, scaler, rf_clf, window)
    if sig is None:
        return None
    anom = window_anomaly(sig, threshold, mode)

    # Bounded, window-scaled decision range. For an attack scenario the verdict
    # is read from windows whose last snapshot falls within the detection-latency
    # budget of the attack onset; a longer window is allowed a proportionally
    # later decision (higher latency, more evidence). Normal scenarios (no
    # attack) are checked over the whole run for the false-positive test.
    active = df["attack_active"].to_numpy()
    active_idx = np.where(active == 1)[0]
    in_range = np.ones_like(anom, dtype=bool)
    if len(active_idx) > 0:
        attack_start = int(active_idx[0])
        deadline = attack_start + C.LATENCY_BUDGET_SNAPS + (window - C.WINDOW_LENGTH)
        in_range = (sig["last_idx"] >= attack_start) & (sig["last_idx"] <= deadline)

    anom_in = anom & in_range
    detected = bool(anom_in.any())

    # First-detection time (simulation seconds) and reported class.
    first_time = None
    reported_class = "normal"
    if detected:
        first_w = int(np.argmax(anom_in))
        first_idx = sig["last_idx"][first_w]
        first_time = float(df["sim_time_s"].to_numpy()[first_idx])
        reported_class = str(sig["rf_pred"][first_w])

    # Severity on the most anomalous window (max reconstruction error).
    sev = severity(sig, norm_stats)

    return {
        "label": df["label"].iloc[0],
        "detected": detected,
        "reported_class": reported_class,
        "first_detection_s": first_time,
        "mean_recon_error": float(np.mean(sig["recon"])),
        "max_recon_error": float(np.max(sig["recon"])),
        "severity": sev,
        "rf_pred_attack_window": _attack_window_pred(df, sig),
    }


def _attack_window_pred(df, sig):
    """RF prediction on the last snapshot inside the attack window (for
    attribution scoring against the ground-truth class)."""
    active = df["attack_active"].to_numpy()
    active_idx = np.where(active == 1)[0]
    if len(active_idx) == 0:
        # normal scenario: use the final window
        return str(sig["rf_pred"][-1])
    target = active_idx[-1]
    # nearest evaluated window whose last snapshot <= target
    valid = np.where(sig["last_idx"] <= target)[0]
    w = valid[-1] if len(valid) else len(sig["rf_pred"]) - 1
    return str(sig["rf_pred"][w])


def severity(sig, norm_stats=None):
    """Severity per Eq. 2 with global min-max normalization.

    ``norm_stats`` provides global (min, max) for reconstruction error and
    phase deviation, computed once over all evaluation scenarios. When absent,
    a robust per-scenario fallback is used.
    """
    w = C.SEVERITY_WEIGHTS
    recon_max = float(np.max(sig["recon"]))
    phase_max = float(np.max(sig["phase_dev"]))
    sensor = float(np.max(sig["sensor_persist"]))
    rf_conf = float(np.max(sig["rf_conf"]))

    if norm_stats:
        e = _mm(recon_max, *norm_stats["recon"])
        p = _mm(phase_max, *norm_stats["phase_dev"])
    else:
        e = min(1.0, recon_max / (recon_max + 1.0))
        p = min(1.0, phase_max / 200.0)
    return float(w["recon"] * e + w["phase_dev"] * p
                 + w["sensor"] * sensor + w["rf_conf"] * rf_conf)


def _mm(x, lo, hi):
    if hi <= lo:
        return 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))
