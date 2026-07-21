"""Random Forest attack-family classifier.

Trains a snapshot-level RF classifier (manuscript Section 4.4) on labeled
telemetry snapshots and reports stratified cross-validated F1 plus a held-out
classification report. The training set is drawn from the attack-window
snapshots of the collector so that each class is represented by its engineered
signature.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix, f1_score  # noqa: E402

from detection import config as C  # noqa: E402

RF_PARAMS = dict(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=C.RANDOM_SEED,
    n_jobs=-1,
)


def build_training_frame(scenario_dfs, per_class=C.TRAIN_SAMPLES_PER_CLASS, rng_seed=C.RANDOM_SEED):
    """Assemble a balanced snapshot-level training frame.

    For attack classes, only attack-active snapshots contribute; for normal,
    any snapshot qualifies. ``per_class`` snapshots are sampled per class.
    """
    import pandas as pd

    rng = np.random.default_rng(rng_seed)
    frames = []
    all_rows = pd.concat(scenario_dfs, ignore_index=True)
    for cls in C.CLASSES:
        if cls == "normal":
            pool = all_rows[all_rows.label == "normal"]
        else:
            pool = all_rows[(all_rows.label == cls) & (all_rows.attack_active == 1)]
        if len(pool) == 0:
            continue
        take = min(per_class, len(pool))
        idx = rng.choice(len(pool), size=take, replace=(len(pool) < per_class))
        frames.append(pool.iloc[idx])
    frame = pd.concat(frames, ignore_index=True)
    return frame


def train(train_frame, out_dir=C.MODELS_DIR, features=None, tag=""):
    features = features or C.FEATURES
    C.ensure_dirs()
    X = train_frame[features].to_numpy(dtype="float32")
    y = train_frame["label"].to_numpy()

    clf = RandomForestClassifier(**RF_PARAMS)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=C.RANDOM_SEED)
    cv_f1 = cross_val_score(clf, X, y, cv=skf, scoring="f1_macro")
    clf.fit(X, y)

    suffix = f"_{tag}" if tag else ""
    joblib.dump(clf, os.path.join(out_dir, f"rf{suffix}.pkl"))
    importances = sorted(
        zip(features, clf.feature_importances_), key=lambda t: t[1], reverse=True)
    meta = {
        "params": RF_PARAMS,
        "features": features,
        "classes": list(clf.classes_),
        "cv_f1_macro_mean": float(cv_f1.mean()),
        "cv_f1_macro_std": float(cv_f1.std()),
        "cv_f1_folds": [float(x) for x in cv_f1],
        "feature_importances": [(f, float(i)) for f, i in importances],
        "n_train": int(len(y)),
        "per_class_counts": {c: int((y == c).sum()) for c in clf.classes_},
    }
    with open(os.path.join(out_dir, f"rf_meta{suffix}.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return clf, meta


def evaluate_report(clf, eval_frame, features=None):
    features = features or C.FEATURES
    X = eval_frame[features].to_numpy(dtype="float32")
    y = eval_frame["label"].to_numpy()
    pred = clf.predict(X)
    report = classification_report(y, pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, pred, labels=list(clf.classes_))
    macro_f1 = f1_score(y, pred, average="macro")
    return report, cm, float(macro_f1)


def load(out_dir=C.MODELS_DIR, tag=""):
    suffix = f"_{tag}" if tag else ""
    clf = joblib.load(os.path.join(out_dir, f"rf{suffix}.pkl"))
    with open(os.path.join(out_dir, f"rf_meta{suffix}.json")) as fh:
        meta = json.load(fh)
    return clf, meta
