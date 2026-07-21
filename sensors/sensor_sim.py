"""Cabinet-sensor simulator.

Supplies the physical (cabinet-facing) telemetry channel: door state, vibration
level, cabinet temperature, and power-supply voltage, plus a derived
sensor-anomaly flag. Normal operation keeps the cabinet closed with low
vibration and nominal temperature; physical-intrusion injection opens the door,
raises vibration above the anomaly threshold, and raises temperature above the
normal envelope (manuscript Section 4.1).
"""

import numpy as np

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detection import config as C


def sample_normal(rng: np.random.Generator) -> dict:
    """Nominal closed-cabinet sensor reading."""
    return {
        "door_state": 0,
        "vibration_level": float(rng.uniform(*C.NORMAL_VIBRATION)),
        "cabinet_temperature_c": float(rng.uniform(*C.NORMAL_CABINET_TEMP_C)),
        "power_voltage": float(rng.uniform(*C.NORMAL_POWER_VOLTAGE)),
    }


def sample_intrusion(rng: np.random.Generator) -> dict:
    """Physical-intrusion reading: door open, high vibration, high temperature.

    Values are drawn from ranges (not fixed constants) so that intrusion
    scenarios carry natural variance rather than a single deterministic point.
    """
    return {
        "door_state": 1,
        "vibration_level": float(rng.uniform(0.90, 1.30)),
        "cabinet_temperature_c": float(rng.uniform(45.0, 58.0)),
        # Intrusion often coincides with a small supply perturbation.
        "power_voltage": float(rng.uniform(11.2, 12.6)),
    }


def derive_sensor_anomaly_flag(reading: dict) -> int:
    """Assert the sensor-anomaly (disturbance) alarm.

    The alarm keys on transient physical disturbance -- vibration above the
    threshold or cabinet temperature above the envelope -- which is sporadic
    during an intrusion. It deliberately does NOT key on ``door_state`` alone:
    door_state is a persistent attribution feature, whereas this alarm is the
    intermittent detection trigger, so the two channels stay distinct.
    """
    anomalous = (
        reading["vibration_level"] > C.VIBRATION_ANOMALY_THRESHOLD
        or reading["cabinet_temperature_c"] > C.CABINET_TEMP_ANOMALY_THRESHOLD
    )
    return int(anomalous)
