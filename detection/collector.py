"""GuardBox collector.

Owns SUMO simulation stepping and injects attack behavior internally so that
each scenario produces consistent telemetry (manuscript Section 4.1). At fixed
intervals it records controller state, timing deviation, traffic and emission
measures, TraCI command features, and cabinet sensor values into a per-snapshot
CSV row.

Usage:
    python -m detection.collector --attack timing --seed 0 --out data/run.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if "SUMO_HOME" not in os.environ:
    try:
        import sumo

        os.environ["SUMO_HOME"] = sumo.__path__[0]
    except Exception:  # pragma: no cover
        pass
sys.path.append(os.path.join(os.environ.get("SUMO_HOME", ""), "tools"))

import traci  # noqa: E402

from detection import config as C  # noqa: E402
from sensors import sensor_sim  # noqa: E402
from attacks import injectors  # noqa: E402


def _sumo_binary():
    home = os.environ["SUMO_HOME"]
    return os.path.join(home, "bin", "sumo")


def _network_totals(monitored_tl):
    """Aggregate base telemetry from SUMO for the current simulation state."""
    veh_ids = traci.vehicle.getIDList()
    waiting = sum(1 for v in veh_ids if traci.vehicle.getSpeed(v) < 0.1)
    speeds = [traci.vehicle.getSpeed(v) for v in veh_ids]
    avg_speed = float(np.mean(speeds)) if speeds else 0.0
    co2 = sum(traci.vehicle.getCO2Emission(v) for v in veh_ids)   # mg/s
    nox = sum(traci.vehicle.getNOxEmission(v) for v in veh_ids)   # mg/s
    phase = traci.trafficlight.getPhase(monitored_tl)
    return {
        "current_phase": int(phase),
        "waiting_vehicles": int(waiting),
        "avg_speed": avg_speed,
        "co2_mg_s": float(co2),
        "nox_mg_s": float(nox),
    }


def collect_scenario(sumocfg, attack_kind="none", seed=0,
                     warmup=C.WARMUP_SNAPSHOTS, max_snaps=C.MAX_SNAPSHOTS,
                     attack_start=C.ATTACK_START_SNAPSHOT,
                     attack_duration=C.ATTACK_DURATION_SNAPSHOTS):
    """Run one scenario and return a DataFrame of per-snapshot telemetry."""
    rng = np.random.default_rng(seed)
    steps_per_snap = max(1, int(round(C.SNAPSHOT_INTERVAL_S / C.STEP_LENGTH_S)))

    sumo_cmd = [
        _sumo_binary(), "-c", sumocfg,
        "--step-length", str(C.STEP_LENGTH_S),
        "--no-warnings", "true", "--no-step-log", "true",
        "--time-to-teleport", "-1",
        "--end", str((warmup + max_snaps + 5) * C.SNAPSHOT_INTERVAL_S),
    ]
    label = C.ATTACK_TO_CLASS[attack_kind]
    from sumolib.miscutils import getFreeSocketPort

    traci.start(sumo_cmd, port=getFreeSocketPort())
    try:
        tls = traci.trafficlight.getIDList()
        monitored_tl = tls[0]

        # Warm-up (discarded).
        for _ in range(warmup * steps_per_snap):
            traci.simulationStep()

        # Track a rolling baseline of command volume for cmd_rate_delta.
        recent_cmd = []
        rows = []
        co2_baseline = None
        for k in range(max_snaps):
            for _ in range(steps_per_snap):
                traci.simulationStep()
            base = _network_totals(monitored_tl)

            # Nominal command channel (observe + occasional set).
            cmd_count = int(rng.integers(*C.NORMAL_CMD_COUNT))
            set_count = int(rng.integers(C.NORMAL_SET_COUNT[0], C.NORMAL_SET_COUNT[1] + 1))
            recent_cmd.append(cmd_count)
            recent_cmd = recent_cmd[-6:]
            cmd_rate_delta = float(cmd_count - np.mean(recent_cmd))

            # Nominal cabinet sensors.
            reading = sensor_sim.sample_normal(rng)

            # Phase deviation vs the expected schedule (normal ~ small noise).
            elapsed = C.EXPECTED_PHASE_DURATION_S + float(rng.normal(0.0, 2.0))
            phase_dev = max(0.0, 100.0 * (elapsed - C.EXPECTED_PHASE_DURATION_S)
                            / C.EXPECTED_PHASE_DURATION_S)

            # CO2 spike flag relative to an early baseline.
            if co2_baseline is None and k >= 3:
                co2_baseline = np.mean([r["co2_mg_s"] for r in rows[:3]]) + 1e-6
            co2_spike = 0
            if co2_baseline:
                co2_spike = int(base["co2_mg_s"] > 2.0 * co2_baseline)

            snap = {
                "snapshot": k,
                "sim_time_s": (warmup + k) * C.SNAPSHOT_INTERVAL_S,
                "current_phase": base["current_phase"],
                "elapsed_seconds": elapsed,
                "phase_deviation_pct": phase_dev,
                "waiting_vehicles": base["waiting_vehicles"],
                "avg_speed": base["avg_speed"],
                "co2_mg_s": base["co2_mg_s"],
                "nox_mg_s": base["nox_mg_s"],
                "traci_cmd_count": cmd_count,
                "traci_set_count_60s": set_count,
                "source_connection_count": C.NORMAL_SOURCE_CONNS,
                "door_state": reading["door_state"],
                "vibration_level": reading["vibration_level"],
                "cabinet_temperature_c": reading["cabinet_temperature_c"],
                "power_voltage": reading["power_voltage"],
                "cmd_rate_delta": cmd_rate_delta,
                "phase_extension_detected": 0,
                "sensor_anomaly_flag": 0,
                "co2_spike_flag": co2_spike,
            }

            in_window = attack_start <= k < (attack_start + attack_duration)
            if attack_kind != "none" and in_window:
                snap = injectors.apply(attack_kind, snap, rng)

            snap["attack_active"] = int(attack_kind != "none" and in_window)
            snap["label"] = label
            snap["attack_kind"] = attack_kind
            rows.append(snap)

        df = pd.DataFrame(rows)
        return df
    finally:
        traci.close()


def main():
    ap = argparse.ArgumentParser(description="GuardBox telemetry collector")
    ap.add_argument("--sumocfg", default=C.SUMO_CONFIGS[C.DEFAULT_TOPOLOGY])
    ap.add_argument("--attack", default="none", choices=C.ATTACK_KINDS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=C.WARMUP_SNAPSHOTS)
    ap.add_argument("--max-steps", type=int, default=C.MAX_SNAPSHOTS)
    ap.add_argument("--attack-start", type=int, default=C.ATTACK_START_SNAPSHOT)
    ap.add_argument("--attack-duration", type=int, default=C.ATTACK_DURATION_SNAPSHOTS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = collect_scenario(
        args.sumocfg, attack_kind=args.attack, seed=args.seed,
        warmup=args.warmup, max_snaps=args.max_steps,
        attack_start=args.attack_start, attack_duration=args.attack_duration,
    )
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"wrote {len(df)} snapshots -> {args.out}")
    else:
        print(df.head())


if __name__ == "__main__":
    main()
