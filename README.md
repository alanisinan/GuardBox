# GuardBox

GuardBox is a multi-modal anomaly-detection framework for intelligent traffic signal
controllers. It monitors controller behavior, TraCI command activity, traffic/emission
variables, and cabinet-style sensor telemetry in a SUMO simulation, and it combines an LSTM
autoencoder (temporal anomaly detection) with a random forest classifier (attack-family
attribution) to detect and label attacks against signal-control logic.

This repository contains the full, runnable pipeline behind the paper *"Multimodal detection
of cyber-physical attacks against intelligent traffic signal controllers"* and reproduces
every table and figure in the manuscript.

## Attack families

Timing manipulation, replay attack, physical intrusion, and combined cyber-physical attack,
plus normal operation. Denial-of-service flooding is out of scope for the current telemetry.

## Repository layout

```
detection/      config, collector, LSTM/RF trainers, detector/fusion, windowing
attacks/        attack-family signature injectors
sensors/        cabinet-sensor simulator
experiments/    evaluation, baselines, window study, multi-intersection, runtime, macro emit
figures/        clean figure regeneration (Figs 2–6)
data/scenarios/ cached per-scenario telemetry CSVs (generated)
results/        result JSONs (generated)
models/         trained models (generated)
*.net.xml, *.rou.xml, *.sumocfg.xml   SUMO topologies (intersection, better, complex)
```

## Requirements

- Python 3.11
- SUMO with TraCI. The simplest cross-platform option is the pip wheel, which bundles the
  `sumo` binary:

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins `eclipse-sumo` (the SUMO binary + tools), TensorFlow, scikit-learn,
XGBoost, pandas, numpy, matplotlib, and seaborn. The pipeline sets `SUMO_HOME` automatically
from the installed `eclipse-sumo` wheel; if you use a system SUMO instead, export `SUMO_HOME`
and add `$SUMO_HOME/tools` to `PYTHONPATH`.

## Reproduce the paper

Run the full experiment suite (generates scenarios on first run, then trains and evaluates):

```bash
python -m experiments.run_all          # core eval + baselines + window + runtime + multi
python -m experiments.emit_latex_macros # write LaTeX result macros/tables
python -m figures.make_figures          # regenerate Figures 2–6
```

Individual stages:

```bash
python -m experiments.run_evaluation --topo better_intersection --n-eval 75
python -m experiments.run_baselines --topo better_intersection --n-eval 75
python -m experiments.run_window_study --windows 12 18 --n-eval 75
python -m experiments.run_multi_intersection --n-eval 25
python -m experiments.run_runtime
```

Collect a single scenario:

```bash
python -m detection.collector --attack timing --seed 0 --out data/example.csv
```

## Protocol

Each scenario samples one telemetry snapshot every 5 simulation seconds: 60 warm-up snapshots
(discarded), a 120-snapshot recorded window, attack onset at snapshot 30 (simulation time
450 s), duration 60 snapshots. Scenarios are partitioned by role into normal-train,
normal-calibration, RF-train, and held-out live-evaluation splits; no live-evaluation scenario
is used to fit, calibrate, or train any model, so the reported false-positive and detection
rates are held-out estimates. (Because the SUMO base environment is deterministic, the
pre-attack normal telemetry of two scenarios can coincide, but no evaluation snapshot ever
enters model fitting as a labeled example.) Every scenario is reproducible from its seed.

## Reproducibility notes

All randomness is seeded (`RANDOM_SEED = 42`; per-scenario seeds via the seed bands in
`experiments/datagen.py`). Results are written as JSON under `results/`; the manuscript's
numbers are emitted as LaTeX macros/tables by `experiments/emit_latex_macros.py`.
