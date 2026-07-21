"""GuardBox detector benchmark, run INSIDE a resource-capped Linux container.

Reports the enforced cgroup limits (proof of the constraint), then loads the
deployed models and measures per-decision inference time and peak memory.
"""
import os, time, resource, glob
import numpy as np
import pandas as pd
import joblib

FEATURES = ["current_phase","elapsed_seconds","phase_deviation_pct","waiting_vehicles",
    "avg_speed","co2_mg_s","nox_mg_s","traci_cmd_count","traci_set_count_60s",
    "source_connection_count","door_state","vibration_level","cabinet_temperature_c",
    "power_voltage","cmd_rate_delta","phase_extension_detected","sensor_anomaly_flag",
    "co2_spike_flag"]
PHYSICAL = ["door_state","vibration_level","cabinet_temperature_c","power_voltage","sensor_anomaly_flag"]
LSTM_FEATURES = [f for f in FEATURES if f not in PHYSICAL]  # 13
WINDOW = 12

def _read(p, default="?"):
    try:
        return open(p).read().strip()
    except Exception:
        return default

print("=== enforced container limits (cgroup v2) ===")
mem_max = _read("/sys/fs/cgroup/memory.max")
cpu_max = _read("/sys/fs/cgroup/cpu.max")
try:
    mem_gb = int(mem_max)/1e9
    print(f"memory.max = {mem_max} ({mem_gb:.2f} GB)")
except ValueError:
    print(f"memory.max = {mem_max}")
q = cpu_max.split()
if len(q) == 2 and q[0] != "max":
    print(f"cpu.max = {cpu_max}  ->  {int(q[0])/int(q[1]):.2f} CPUs (quota/period)")
else:
    print(f"cpu.max = {cpu_max}")
print(f"os.cpu_count() = {os.cpu_count()}")

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(1)
from tensorflow import keras

print("\n=== loading deployed models ===")
lstm = keras.models.load_model("lstm_ae_better_intersection.keras")
scaler = joblib.load("scaler_better_intersection.pkl")
rf = joblib.load("rf_better_intersection.pkl")
print(f"LSTM params = {lstm.count_params():,}; RF trees = {len(rf.estimators_)}")

df = pd.read_csv(glob.glob("*.csv")[0])
win = df[LSTM_FEATURES].to_numpy(dtype="float32")[:WINDOW]
scaled = scaler.transform(win).reshape(1, WINDOW, len(LSTM_FEATURES)).astype("float32")
snap = df[FEATURES].to_numpy(dtype="float32")[:1]

lstm.predict(scaled, verbose=0); rf.predict(snap)  # warm up
N = 200
t = time.time()
for _ in range(N):
    lstm.predict(scaled, verbose=0)
    rf.predict(snap)
ms = (time.time() - t) / N * 1e3
peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # Linux: KB -> GB

print("\n=== RESULT (inside the capped container) ===")
print(f"decisions timed: {N}")
print(f"mean decision time: {ms:.1f} ms  (LSTM reconstruction + RF classification)")
print(f"peak resident memory: {peak_gb:.2f} GB")
print(f"time budget headroom vs 5 s snapshot interval: {5000/ms:.0f}x")
