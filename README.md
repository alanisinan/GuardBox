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

`emit_latex_macros` writes `gb_results.tex` and the `gb_tab_*.tex` tables to `latex/`; set
`GUARDBOX_LATEX_DIR` to write them straight into a manuscript project folder instead.

Individual stages:

```bash
python -m experiments.run_evaluation --topo better_intersection --n-eval 75
python -m experiments.run_baselines --topo better_intersection --n-eval 75
python -m experiments.run_window_study --windows 12 18 --n-eval 75
python -m experiments.run_multi_intersection --n-eval 25
python -m experiments.run_runtime
python -m experiments.run_robustness
python -m experiments.run_init_sensitivity   # after run_evaluation + run_robustness
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

All randomness is seeded (`RANDOM_SEED = 42` in `detection/config.py`; per-scenario seeds via
the seed bands in `experiments/datagen.py`). The Keras models (the LSTM autoencoder and the
1D-CNN/GRU baselines) are seeded with `tf.keras.utils.set_random_seed`, which fixes Python,
NumPy, and TensorFlow randomness together, so retraining is deterministic for a given platform
and TensorFlow version. Results
are written as JSON under `results/`; the manuscript's numbers are emitted as LaTeX
macros/tables by `experiments/emit_latex_macros.py`.

The published results under `results/` were produced before that seeding change, when
`detection/train_lstm.py` called `tf.random.set_seed` only. Under Keras 3 this fixes
TensorFlow's global seed but not the layer weight initializers, so a re-run starts from a
different initialization than the published model. `experiments/run_init_sensitivity.py`
measures what that changes: retraining under seeds 42, 1, 2, and 3 reproduces every detection
count, Wilson interval, ablation entry, detection-latency bound, and attribution score of the
main evaluation exactly, and rerunning the baseline, window-length, and multi-intersection
studies with the current scripts reproduces the values the paper reports for them exactly. The
continuous LSTM-derived values (mean reconstruction errors, the separation ratio, severities)
shift slightly, and the noisy-normal false-positive rates of the robustness study at σ = 0.25
and 0.5 vary with the initialization. Per-seed values are in `results/init_sensitivity.json`.

## License

Released under the MIT License (see `LICENSE`). The license covers the code, the SUMO
configuration files, and the data and results in this repository.
