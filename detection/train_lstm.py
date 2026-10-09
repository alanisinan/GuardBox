"""LSTM autoencoder training and threshold calibration.

Trains an encoder-decoder LSTM autoencoder on normal telemetry windows only,
using a strict scenario-level split so that no scenario contributes windows to
more than one of {train, calibration, evaluation}. The reconstruction-error
threshold is calibrated on held-out normal windows.

Architecture and hyperparameters are dumped to models/lstm_meta.json so the
manuscript can report them in full.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import joblib  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from detection import config as C  # noqa: E402
from detection.windowing import scenario_windows  # noqa: E402

# Architecture
ENCODER_UNITS = [64, 32]
LATENT_DIM = 16
DECODER_UNITS = [32, 64]
EPOCHS = 40
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
THRESHOLD_PERCENTILE = 100.0   # max normal calibration error
# Conservative operating point: false dispatches are operationally costly, so the
# threshold is set well above the normal-calibration envelope. This is an a priori
# FP-averse design choice, not a fit to the evaluation set.
THRESHOLD_MARGIN = 1.50        # 50% headroom above the calibration maximum


def build_autoencoder(window, n_features, seed=C.RANDOM_SEED):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    # Seed Python, NumPy and TensorFlow together. Under Keras 3 the layer weight
    # initializers draw on Python's RNG, so tf.random.set_seed alone does not fix
    # them; this makes weight initialization and mini-batch shuffling deterministic.
    tf.keras.utils.set_random_seed(seed)
    m = models.Sequential(name="lstm_autoencoder")
    m.add(layers.Input((window, n_features)))
    m.add(layers.LSTM(ENCODER_UNITS[0], return_sequences=True))
    m.add(layers.LSTM(ENCODER_UNITS[1], return_sequences=False))
    m.add(layers.Dense(LATENT_DIM, activation="relu", name="latent"))
    m.add(layers.RepeatVector(window))
    m.add(layers.LSTM(DECODER_UNITS[0], return_sequences=True))
    m.add(layers.LSTM(DECODER_UNITS[1], return_sequences=True))
    m.add(layers.TimeDistributed(layers.Dense(n_features)))
    m.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE), loss="mse")
    return m


def window_recon_errors(model, windows_scaled):
    """Per-window mean squared reconstruction error."""
    if len(windows_scaled) == 0:
        return np.array([])
    recon = model.predict(windows_scaled, verbose=0)
    return np.mean((windows_scaled - recon) ** 2, axis=(1, 2))


def fit_scaler(normal_dfs, features=None):
    features = features or C.LSTM_FEATURES
    stacked = np.concatenate([df[features].to_numpy(dtype="float32") for df in normal_dfs])
    scaler = StandardScaler().fit(stacked)
    return scaler


def scale_windows(windows, scaler):
    if len(windows) == 0:
        return windows
    n, w, f = windows.shape
    flat = scaler.transform(windows.reshape(-1, f))
    return flat.reshape(n, w, f).astype("float32")


def train(normal_train_dfs, normal_calib_dfs, window=C.WINDOW_LENGTH,
          out_dir=C.MODELS_DIR, features=None, tag="", seed=C.RANDOM_SEED):
    features = features or C.LSTM_FEATURES
    C.ensure_dirs()
    scaler = fit_scaler(normal_train_dfs, features)

    train_windows = np.concatenate(
        [scale_windows(scenario_windows(df, window, features)[0], scaler)
         for df in normal_train_dfs])

    model = build_autoencoder(window, len(features), seed=seed)
    model.fit(train_windows, train_windows, epochs=EPOCHS, batch_size=BATCH_SIZE,
              validation_split=0.1, verbose=0, shuffle=True)

    calib_windows = np.concatenate(
        [scale_windows(scenario_windows(df, window, features)[0], scaler)
         for df in normal_calib_dfs])
    calib_err = window_recon_errors(model, calib_windows)
    threshold = float(np.percentile(calib_err, THRESHOLD_PERCENTILE) * THRESHOLD_MARGIN)

    suffix = f"_{tag}" if tag else ""
    model.save(os.path.join(out_dir, f"lstm_ae{suffix}.keras"))
    joblib.dump(scaler, os.path.join(out_dir, f"scaler{suffix}.pkl"))
    meta = {
        "window_length": window,
        "n_features": len(features),
        "features": features,
        "encoder_units": ENCODER_UNITS,
        "latent_dim": LATENT_DIM,
        "decoder_units": DECODER_UNITS,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "optimizer": "adam",
        "loss": "mse",
        "seed": seed,
        "scaler": "StandardScaler (fit on normal-train only)",
        "threshold": threshold,
        "threshold_rule": f"{THRESHOLD_PERCENTILE:.0f}th pct of normal-calib error x {THRESHOLD_MARGIN}",
        "calib_error_mean": float(np.mean(calib_err)),
        "calib_error_max": float(np.max(calib_err)),
        "n_train_windows": int(len(train_windows)),
        "n_calib_windows": int(len(calib_windows)),
        "total_params": int(model.count_params()),
    }
    with open(os.path.join(out_dir, f"lstm_meta{suffix}.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return model, scaler, threshold, meta


def load(out_dir=C.MODELS_DIR, tag=""):
    import tensorflow as tf

    suffix = f"_{tag}" if tag else ""
    model = tf.keras.models.load_model(os.path.join(out_dir, f"lstm_ae{suffix}.keras"))
    scaler = joblib.load(os.path.join(out_dir, f"scaler{suffix}.pkl"))
    with open(os.path.join(out_dir, f"lstm_meta{suffix}.json")) as fh:
        meta = json.load(fh)
    return model, scaler, meta["threshold"], meta
