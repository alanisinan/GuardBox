"""Scenario generation with an on-disk cache and a strict scenario-level split.

Split design (documented for the manuscript, addresses the train/test question):
    normal_train  : normal scenarios used ONLY to fit the LSTM scaler + weights.
    normal_calib  : disjoint normal scenarios used ONLY to calibrate the LSTM
                    reconstruction-error threshold.
    rf_train      : scenarios (all classes) used ONLY to train the RF.
    eval          : 75-per-class live-evaluation set (Tables 1, 3).
No live-evaluation scenario is used to fit, calibrate, or train any model (the RF
trains only on attack-window snapshots from rf_train; the LSTM only on
normal_train/normal_calib), so the 0% false-positive rate and the detection rates
are held-out estimates. Note: because the SUMO base environment is deterministic,
pre-attack normal telemetry can coincide across scenarios; the splits are disjoint
at the scenario (role) level, not by globally unique random draw.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detection import config as C  # noqa: E402
from detection.collector import collect_scenario  # noqa: E402

SCEN_DIR = os.path.join(C.DATA_DIR, "scenarios")

# Per-split base seed offsets. Scenario identity is (split, class, seed); the
# integer seed values can recur across splits for different classes, but no
# live-evaluation scenario is ever used for training (see module docstring).
SEED_BANDS = {
    "normal_train": 1000,
    "normal_calib": 1500,
    "rf_train": 3000,
    "eval": 5000,
}


def _path(split, cls, seed, topo):
    d = os.path.join(SCEN_DIR, topo, split)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{cls}_{seed}.csv")


def get_scenario(split, attack_kind, seed, topo=C.DEFAULT_TOPOLOGY, force=False):
    cls = C.ATTACK_TO_CLASS[attack_kind]
    path = _path(split, cls, seed, topo)
    if os.path.exists(path) and not force:
        return pd.read_csv(path)
    sumocfg = C.SUMO_CONFIGS[topo]
    df = collect_scenario(sumocfg, attack_kind=attack_kind, seed=seed)
    df.to_csv(path, index=False)
    return df


# Per-topology seed offset so that different topologies draw independent random
# scenarios rather than reusing identical RNG streams. The primary topology keeps
# offset 0 so its cached 375-scenario evaluation is unchanged.
TOPO_SEED_OFFSET = {
    "better_intersection": 0,
    "intersection": 300000,
    "complex_intersection": 600000,
}


def generate_split(split, kinds, n_per, topo=C.DEFAULT_TOPOLOGY, progress=True):
    base = SEED_BANDS[split] + TOPO_SEED_OFFSET.get(topo, 0)
    dfs = []
    for ki, kind in enumerate(kinds):
        for i in range(n_per):
            seed = base + ki * 1000 + i
            dfs.append(get_scenario(split, kind, seed, topo))
        if progress:
            print(f"  [{split}] {C.ATTACK_TO_CLASS[kind]}: {n_per} scenarios", flush=True)
    return dfs


def build_all(topo=C.DEFAULT_TOPOLOGY,
              n_eval=C.SCENARIOS_PER_CLASS, n_rf_train=10,
              n_normal_train=60, n_normal_calib=20, progress=True):
    """Generate (or load) every split for a topology. Returns a dict of lists."""
    if progress:
        print(f"[datagen] topology={topo}", flush=True)
    normal_train = generate_split("normal_train", ["none"], n_normal_train, topo, progress)
    normal_calib = generate_split("normal_calib", ["none"], n_normal_calib, topo, progress)
    rf_train = generate_split("rf_train", C.ATTACK_KINDS, n_rf_train, topo, progress)
    eval_dfs = generate_split("eval", C.ATTACK_KINDS, n_eval, topo, progress)
    return {
        "normal_train": normal_train,
        "normal_calib": normal_calib,
        "rf_train": rf_train,
        "eval": eval_dfs,
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--topo", default=C.DEFAULT_TOPOLOGY, choices=list(C.SUMO_CONFIGS))
    ap.add_argument("--n-eval", type=int, default=C.SCENARIOS_PER_CLASS)
    args = ap.parse_args()
    build_all(topo=args.topo, n_eval=args.n_eval)
    print("done")
