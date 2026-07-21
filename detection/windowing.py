"""Shared windowing helpers for building LSTM sequences from telemetry."""

import numpy as np

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detection import config as C


def scenario_windows(df, window=C.WINDOW_LENGTH, features=None):
    """Return sliding windows of shape (n_windows, window, n_features) and the
    index of the last snapshot in each window."""
    features = features or C.FEATURES
    arr = df[features].to_numpy(dtype="float32")
    n = arr.shape[0]
    if n < window:
        return np.empty((0, window, arr.shape[1]), dtype="float32"), np.array([], dtype=int)
    windows, last_idx = [], []
    for end in range(window, n + 1):
        windows.append(arr[end - window:end])
        last_idx.append(end - 1)
    return np.stack(windows), np.array(last_idx, dtype=int)
