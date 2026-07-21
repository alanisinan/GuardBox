"""Attack injectors.

Each injector perturbs the feature signature of a telemetry snapshot to encode
the operational definition of one attack family (manuscript Section 4.5). The
collector calls these from inside the simulation loop during the attack window,
so attack timing is aligned with telemetry collection by construction.

Design note (transparency for the separability discussion): attack signatures
are injected as controlled feature perturbations layered on a SUMO-derived
normal baseline. Perturbations are drawn from ranges (with per-snapshot noise),
not fixed constants, so each family carries natural within-class variance.

Signature summary
-----------------
timing   : phase deviation > 100%, phase-extension flag set, moderate set
           commands, no sensor anomaly.
replay   : phase deviation near zero, no extension, sharp command / set-command
           burst, high command-rate delta, no sensor anomaly.
physical : phase deviation zero, no extension, cabinet sensor anomaly asserted
           (door open, vibration > 0.9, temperature > 45 C).
combined : high phase deviation AND cabinet sensor anomaly simultaneously, with
           moderate command-rate effects.
"""

import numpy as np

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detection import config as C
from sensors import sensor_sim


def inject_timing(snap: dict, rng: np.random.Generator) -> dict:
    """Timing manipulation: extend green phase beyond the expected schedule."""
    snap["phase_deviation_pct"] = float(rng.uniform(100.0, 165.0))
    snap["phase_extension_detected"] = 1
    snap["elapsed_seconds"] = float(
        C.EXPECTED_PHASE_DURATION_S * (1.0 + snap["phase_deviation_pct"] / 100.0)
    )
    # A phase override is a set command; timing attack issues a few of them.
    snap["traci_set_count_60s"] = int(rng.integers(3, 8))
    snap["traci_cmd_count"] = snap["traci_set_count_60s"] + int(rng.integers(2, 6))
    return snap


def inject_replay(snap: dict, rng: np.random.Generator) -> dict:
    """Replay attack: burst of repeated previously-valid set commands."""
    snap["phase_deviation_pct"] = float(abs(rng.normal(0.0, 1.5)))
    snap["phase_extension_detected"] = 0
    burst = int(rng.integers(60, 140))          # repeated command burst
    snap["traci_set_count_60s"] = burst
    snap["traci_cmd_count"] = burst + int(rng.integers(20, 60))
    snap["cmd_rate_delta"] = float(rng.uniform(40.0, 120.0))
    # Replay may originate from an additional injected session.
    snap["source_connection_count"] = int(rng.integers(2, 4))
    return snap


def inject_physical(snap: dict, rng: np.random.Generator) -> dict:
    """Physical intrusion: intermittent cabinet sensor anomaly, timing untouched.

    The intrusion signal is sporadic: on each snapshot the cabinet sensors show
    an anomalous reading with probability INTRUSION_FLAG_PROB and a near-normal
    reading otherwise. This intermittency is what makes the physical channel
    depend on how much flagged evidence a rolling window accumulates.
    """
    snap["phase_deviation_pct"] = float(abs(rng.normal(0.0, 1.5)))
    snap["phase_extension_detected"] = 0
    reading = _intrusion_reading(rng)
    snap.update(reading)
    snap["sensor_anomaly_flag"] = sensor_sim.derive_sensor_anomaly_flag(reading)
    return snap


def _intrusion_reading(rng: np.random.Generator) -> dict:
    """Cabinet reading during an intrusion: door open persistently, but the
    vibration/temperature disturbance (hence the sensor-anomaly alarm) is
    sporadic -- anomalous with probability INTRUSION_FLAG_PROB, near-normal
    otherwise."""
    if rng.random() < C.INTRUSION_FLAG_PROB:
        reading = sensor_sim.sample_intrusion(rng)
    else:
        reading = sensor_sim.sample_normal(rng)
    reading["door_state"] = 1        # cabinet stays open throughout the intrusion
    return reading


def inject_combined(snap: dict, rng: np.random.Generator) -> dict:
    """Combined attack: timing deviation AND cabinet sensor anomaly together."""
    snap["phase_deviation_pct"] = float(rng.uniform(100.0, 165.0))
    snap["phase_extension_detected"] = 1
    snap["elapsed_seconds"] = float(
        C.EXPECTED_PHASE_DURATION_S * (1.0 + snap["phase_deviation_pct"] / 100.0)
    )
    snap["traci_set_count_60s"] = int(rng.integers(4, 12))
    snap["traci_cmd_count"] = snap["traci_set_count_60s"] + int(rng.integers(3, 10))
    snap["cmd_rate_delta"] = float(rng.uniform(5.0, 25.0))
    reading = _intrusion_reading(rng)
    snap.update(reading)
    snap["sensor_anomaly_flag"] = sensor_sim.derive_sensor_anomaly_flag(reading)
    return snap


INJECTORS = {
    "timing": inject_timing,
    "replay": inject_replay,
    "physical": inject_physical,
    "combined": inject_combined,
}


def apply(attack_kind: str, snap: dict, rng: np.random.Generator) -> dict:
    if attack_kind in (None, "none"):
        return snap
    return INJECTORS[attack_kind](snap, rng)
