"""Shared configuration and constants for the GuardBox detection pipeline.

This module centralizes every tunable used across the collector, the attack
injectors, the LSTM autoencoder, the Random Forest classifier, and the detector,
so that a single source of truth backs every number reported in the manuscript.

Protocol / timing note (addresses the detection-latency units question):
    The collector samples one telemetry snapshot every ``SNAPSHOT_INTERVAL_S``
    seconds of simulation time. A run consists of ``WARMUP_SNAPSHOTS`` warm-up
    snapshots (discarded) followed by ``MAX_SNAPSHOTS`` recorded snapshots. An
    attack begins at recorded-snapshot index ``ATTACK_START_SNAPSHOT`` and lasts
    ``ATTACK_DURATION_SNAPSHOTS`` snapshots. The simulation time of the attack
    onset is therefore
        t_attack = (WARMUP_SNAPSHOTS + ATTACK_START_SNAPSHOT) * SNAPSHOT_INTERVAL_S
                 = (60 + 30) * 5 = 450 s,
    which is why archived first-detection times fall in the 435-500 s band.
"""

# ---------------------------------------------------------------------------
# Experiment protocol
# ---------------------------------------------------------------------------
SNAPSHOT_INTERVAL_S = 5          # simulation seconds between telemetry snapshots
STEP_LENGTH_S = 1.0              # SUMO step length (standardized across topologies)
WARMUP_SNAPSHOTS = 60            # warm-up snapshots (excluded from recording)
MAX_SNAPSHOTS = 120              # recorded snapshots per scenario
ATTACK_START_SNAPSHOT = 30       # attack onset, index into the recorded window
ATTACK_DURATION_SNAPSHOTS = 60   # attack length in snapshots

SCENARIOS_PER_CLASS = 75         # live-evaluation scenarios per class
TRAIN_SAMPLES_PER_CLASS = 200    # labeled snapshots per class for RF training set

# ---------------------------------------------------------------------------
# Sequence / model
# ---------------------------------------------------------------------------
WINDOW_LENGTH = 12               # LSTM rolling-window length (snapshots)
RANDOM_SEED = 42

# Expected (nominal) green-phase schedule used to compute phase deviation.
EXPECTED_PHASE_DURATION_S = 30.0

# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------
CLASSES = [
    "normal",
    "timing_manipulation",
    "replay_attack",
    "physical_intrusion",
    "combined_attack",
]
ATTACK_KINDS = ["none", "timing", "replay", "physical", "combined"]
ATTACK_TO_CLASS = {
    "none": "normal",
    "timing": "timing_manipulation",
    "replay": "replay_attack",
    "physical": "physical_intrusion",
    "combined": "combined_attack",
}

# ---------------------------------------------------------------------------
# Feature schema (18 monitored features, manuscript Section 4.2)
# ---------------------------------------------------------------------------
FEATURES = [
    "current_phase",
    "elapsed_seconds",
    "phase_deviation_pct",
    "waiting_vehicles",
    "avg_speed",
    "co2_mg_s",
    "nox_mg_s",
    "traci_cmd_count",
    "traci_set_count_60s",
    "source_connection_count",
    "door_state",
    "vibration_level",
    "cabinet_temperature_c",
    "power_voltage",
    "cmd_rate_delta",
    "phase_extension_detected",
    "sensor_anomaly_flag",
    "co2_spike_flag",
]

# Physical cabinet-sensor channel. These are handled as a SEPARATE direct
# detection channel (sensor-anomaly gate), NOT fed to the LSTM autoencoder, so
# that a physical intrusion which leaves controller/traffic behavior unchanged
# does not raise LSTM reconstruction error. The RF classifier still uses the
# full feature set for attribution.
PHYSICAL_FEATURES = [
    "door_state",
    "vibration_level",
    "cabinet_temperature_c",
    "power_voltage",
    "sensor_anomaly_flag",
]
# Behavioral/command/traffic features the LSTM autoencoder reconstructs.
LSTM_FEATURES = [f for f in FEATURES if f not in PHYSICAL_FEATURES]

# ---------------------------------------------------------------------------
# Cabinet-sensor normal envelopes (physical channel)
# ---------------------------------------------------------------------------
NORMAL_VIBRATION = (0.02, 0.15)          # g, low ambient vibration
NORMAL_CABINET_TEMP_C = (18.0, 32.0)     # deg C nominal envelope
NORMAL_POWER_VOLTAGE = (11.8, 12.4)      # V, nominal 12 V supply
VIBRATION_ANOMALY_THRESHOLD = 0.9        # g
CABINET_TEMP_ANOMALY_THRESHOLD = 45.0    # deg C

# ---------------------------------------------------------------------------
# Normal command-channel envelope (per snapshot)
# ---------------------------------------------------------------------------
NORMAL_CMD_COUNT = (2, 6)                # observe/set commands per snapshot
NORMAL_SET_COUNT = (0, 2)
NORMAL_SOURCE_CONNS = 1                  # single trusted controller session

# ---------------------------------------------------------------------------
# Detection thresholds (calibrated in train_lstm.py; defaults are fallbacks)
# ---------------------------------------------------------------------------
# Physical channel gate. The sensor-anomaly flag is asserted intermittently
# during a physical intrusion (sporadic vibration / contact bounce), so the
# physical channel fires only when a rolling window accumulates at least
# SENSOR_PERSIST_COUNT flagged snapshots. Because this is an absolute evidence
# count, a longer window (e.g. 18) accumulates more flagged snapshots before the
# decision deadline and therefore raises physical-intrusion detection, at the
# cost of higher latency -- the trade-off studied in run_window_study.py.
SENSOR_PERSIST_COUNT = 6
INTRUSION_FLAG_PROB = 0.6        # per-snapshot probability the flag is asserted

# Bounded detection latency: a scenario's verdict is read from rolling windows
# whose last snapshot falls within LATENCY_BUDGET_SNAPS of the attack onset,
# plus (window - 12) extra snapshots for longer windows (later decision).
LATENCY_BUDGET_SNAPS = 10

# Severity weights (manuscript Eq. 2)
SEVERITY_WEIGHTS = {"recon": 0.4, "phase_dev": 0.3, "sensor": 0.2, "rf_conf": 0.1}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
MODELS_DIR = os.path.join(REPO_ROOT, "models")
FIG_DIR = os.path.join(REPO_ROOT, "figures")

SUMO_CONFIGS = {
    "intersection": "intersection.sumocfg.xml",
    "better_intersection": "better_intersection.sumocfg.xml",
    "complex_intersection": "complex_intersection.sumocfg.xml",
}
DEFAULT_TOPOLOGY = "better_intersection"


def ensure_dirs():
    for d in (DATA_DIR, RESULTS_DIR, MODELS_DIR, FIG_DIR):
        os.makedirs(d, exist_ok=True)
