"""Baseline comparison (editor's explicit request; Reviewers 1 and 2).

Compares GuardBox's RF attribution stage and its LSTM detection stage against
standard alternatives on the SAME telemetry, features, and strict scenario-level
split:

  Attribution (5-class):
      snapshot classifiers: RF (GuardBox) | SVM-RBF | XGBoost | MLP | Logistic Regression
      deep-learning models over the 12-snapshot window: 1D-CNN | GRU
  Detection (unsupervised anomaly, trained on normal only):
      LSTM autoencoder (GuardBox) | Isolation Forest | One-Class SVM

Writes results/baselines_<topo>.json.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from sklearn.svm import SVC, OneClassSVM  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.ensemble import RandomForestClassifier, IsolationForest  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.metrics import f1_score, accuracy_score  # noqa: E402
import xgboost as xgb  # noqa: E402

from detection import config as C, train_rf, train_lstm, detector  # noqa: E402
from experiments import datagen  # noqa: E402


def attribution_baselines(rf_frame, eval_frame, features):
    X_tr = rf_frame[features].to_numpy(dtype="float32")
    y_tr = rf_frame["label"].to_numpy()
    X_te = eval_frame[features].to_numpy(dtype="float32")
    y_te = eval_frame["label"].to_numpy()

    models = {
        "RandomForest (GuardBox)": RandomForestClassifier(**train_rf.RF_PARAMS),
        "SVM-RBF": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale",
                                                       class_weight="balanced", random_state=C.RANDOM_SEED)),
        "XGBoost": xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                     subsample=0.9, colsample_bytree=0.9,
                                     random_state=C.RANDOM_SEED, tree_method="hist",
                                     eval_metric="mlogloss"),
        "MLP": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32),
                             max_iter=500, random_state=C.RANDOM_SEED)),
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=C.RANDOM_SEED)),
    }
    out = {}
    for name, model in models.items():
        if name == "XGBoost":
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder().fit(y_tr)
            t = time.time()
            model.fit(X_tr, le.transform(y_tr))
            pred = le.inverse_transform(model.predict(X_te))
            fit_ms = (time.time() - t) * 1e3
        else:
            t = time.time()
            model.fit(X_tr, y_tr)
            pred = model.predict(X_te)
            fit_ms = (time.time() - t) * 1e3
        out[name] = {
            "macro_f1": float(f1_score(y_te, pred, average="macro", labels=C.CLASSES)),
            "accuracy": float(accuracy_score(y_te, pred)),
            "per_class_f1": {c: float(f1_score(y_te == c, pred == c, zero_division=0))
                             for c in C.CLASSES},
            "fit_ms": fit_ms,
        }
    return out


def detection_baselines(normal_train_dfs, eval_dfs, features):
    """Unsupervised detectors trained on normal snapshots only.

    For a fair comparison with GuardBox, each baseline uses the SAME scenario
    decision rule: a scenario is flagged when a rolling window accumulates at
    least SENSOR_PERSIST_COUNT anomalous snapshots (evidence persistence), and
    for attack scenarios the decision is read within the bounded latency window.
    A low, fixed contamination is used because the training data is all-normal.
    """
    import pandas as pd

    norm = pd.concat(normal_train_dfs, ignore_index=True)[features].to_numpy(dtype="float32")
    scaler = StandardScaler().fit(norm)
    norm_s = scaler.transform(norm)

    iso = IsolationForest(n_estimators=200, contamination=0.01,
                          random_state=C.RANDOM_SEED).fit(norm_s)
    ocsvm = OneClassSVM(kernel="rbf", nu=0.01, gamma="scale").fit(norm_s)

    W = C.WINDOW_LENGTH
    K = C.SENSOR_PERSIST_COUNT

    def scenario_detected(model, df):
        rows = df[features].to_numpy(dtype="float32")
        anom = (model.predict(scaler.transform(rows)) == -1).astype(int)  # per-snapshot
        n = len(rows)
        # windowed evidence-persistence, mirroring the GuardBox physical gate
        fire = np.array([anom[e - W:e].sum() >= K for e in range(W, n + 1)])
        last_idx = np.arange(W - 1, n)
        active = df["attack_active"].to_numpy()
        aidx = np.where(active == 1)[0]
        if len(aidx) > 0:
            a0 = int(aidx[0])
            dl = a0 + C.LATENCY_BUDGET_SNAPS
            in_range = (last_idx >= a0) & (last_idx <= dl)
        else:
            in_range = np.ones_like(fire, dtype=bool)  # whole run for normal (FP test)
        return int((fire & in_range).any())

    def scenario_rate(model):
        counts = {c: {"n": 0, "det": 0} for c in C.CLASSES}
        for df in eval_dfs:
            cls = df["label"].iloc[0]
            counts[cls]["n"] += 1
            counts[cls]["det"] += scenario_detected(model, df)
        return {c: counts[c]["det"] / counts[c]["n"] if counts[c]["n"] else 0.0
                for c in C.CLASSES}

    return {"IsolationForest": scenario_rate(iso), "OneClassSVM": scenario_rate(ocsvm)}


def _windowed(dfs, features, W, eval_one, per_class=None, seed=C.RANDOM_SEED):
    """Build (n, W, n_features) temporal windows labeled by scenario class.

    For training (eval_one=False), one window per attack-active snapshot (any
    snapshot for normal). For evaluation (eval_one=True), one window per scenario
    ending at its last attack-window snapshot (last snapshot for normal), so the
    deep models are scored per scenario on the same live-evaluation set.
    """
    rng = np.random.default_rng(seed)
    per_cls = {c: [] for c in C.CLASSES}
    for df in dfs:
        cls = df["label"].iloc[0]
        arr = df[features].to_numpy(dtype="float32")
        n = len(arr)
        active = df["attack_active"].to_numpy()
        if eval_one:
            aidx = np.where(active == 1)[0]
            t = int(aidx[-1]) if len(aidx) else n - 1
            t = max(t, W - 1)
            per_cls[cls].append(arr[t - W + 1:t + 1])
        else:
            targets = (range(W - 1, n) if cls == "normal"
                       else [i for i in np.where(active == 1)[0] if i >= W - 1])
            for t in targets:
                per_cls[cls].append(arr[t - W + 1:t + 1])
    X, y = [], []
    for c, wins in per_cls.items():
        if not wins:
            continue
        wins = np.stack(wins)
        if per_class and len(wins) > per_class:
            wins = wins[rng.choice(len(wins), per_class, replace=False)]
        X.append(wins)
        y += [c] * len(wins)
    return np.concatenate(X).astype("float32"), np.array(y)


def deep_learning_baselines(rf_train_dfs, eval_dfs, features):
    """Representative deep-learning IDS architectures (1D-CNN, GRU) over the same
    12-snapshot temporal window, trained and scored on the same splits."""
    import tensorflow as tf
    from tensorflow.keras import layers, models

    tf.random.set_seed(C.RANDOM_SEED)
    W = C.WINDOW_LENGTH
    Xtr, ytr = _windowed(rf_train_dfs, features, W, eval_one=False, per_class=200)
    Xte, yte = _windowed(eval_dfs, features, W, eval_one=True)

    # standardize on the flattened training windows
    nf = len(features)
    scaler = StandardScaler().fit(Xtr.reshape(-1, nf))
    Xtr = scaler.transform(Xtr.reshape(-1, nf)).reshape(-1, W, nf).astype("float32")
    Xte = scaler.transform(Xte.reshape(-1, nf)).reshape(-1, W, nf).astype("float32")
    classes = list(C.CLASSES)
    idx = {c: i for i, c in enumerate(classes)}
    ytr_i = np.array([idx[c] for c in ytr])

    def _cnn():
        m = models.Sequential([
            layers.Input((W, nf)),
            layers.Conv1D(32, 3, activation="relu", padding="same"),
            layers.Conv1D(32, 3, activation="relu", padding="same"),
            layers.GlobalMaxPooling1D(),
            layers.Dense(32, activation="relu"),
            layers.Dense(len(classes), activation="softmax")])
        m.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
        return m

    def _gru():
        m = models.Sequential([
            layers.Input((W, nf)),
            layers.GRU(48),
            layers.Dense(32, activation="relu"),
            layers.Dense(len(classes), activation="softmax")])
        m.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
        return m

    out = {}
    for name, build in [("1D-CNN", _cnn), ("GRU", _gru)]:
        model = build(); t = time.time()
        model.fit(Xtr, ytr_i, epochs=40, batch_size=64, verbose=0, shuffle=True)
        pred = np.array(classes)[model.predict(Xte, verbose=0).argmax(1)]
        out[name] = {
            "macro_f1": float(f1_score(yte, pred, average="macro", labels=classes)),
            "accuracy": float(accuracy_score(yte, pred)),
            "per_class_f1": {c: float(f1_score(yte == c, pred == c, zero_division=0)) for c in classes},
            "fit_ms": (time.time() - t) * 1e3,
        }
    return out


def run(topo=C.DEFAULT_TOPOLOGY, n_eval=C.SCENARIOS_PER_CLASS):
    C.ensure_dirs()
    splits = datagen.build_all(topo=topo, n_eval=n_eval)
    features = C.FEATURES

    rf_frame = train_rf.build_training_frame(splits["rf_train"])
    # Eval attribution frame: attack-window snapshots + normal snapshots (balanced-ish).
    import pandas as pd
    eval_rows = pd.concat(splits["eval"], ignore_index=True)
    eval_frame = pd.concat([
        eval_rows[(eval_rows.label == c) & ((eval_rows.attack_active == 1) | (c == "normal"))]
        for c in C.CLASSES
    ], ignore_index=True)

    attribution = attribution_baselines(rf_frame, eval_frame, features)
    attribution.update(deep_learning_baselines(splits["rf_train"], splits["eval"], features))
    detection = detection_baselines(splits["normal_train"], splits["eval"], features)

    results = {"topology": topo, "attribution": attribution, "detection": detection}
    out = os.path.join(C.RESULTS_DIR, f"baselines_{topo}.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[baselines] wrote {out}")
    print("\n=== attribution macro-F1 (5-class) ===")
    for name, m in attribution.items():
        print(f"  {name:26s} F1={m['macro_f1']:.4f}  acc={m['accuracy']:.4f}")
    print("=== unsupervised detection rate (physical intrusion is the hard class) ===")
    for name, r in detection.items():
        print(f"  {name:18s} " + "  ".join(f"{c.split('_')[0]}={r[c]*100:.0f}%" for c in C.CLASSES))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--topo", default=C.DEFAULT_TOPOLOGY, choices=list(C.SUMO_CONFIGS))
    ap.add_argument("--n-eval", type=int, default=C.SCENARIOS_PER_CLASS)
    args = ap.parse_args()
    run(topo=args.topo, n_eval=args.n_eval)
