# Constrained-hardware deployment benchmark

Runs the deployed GuardBox detector inside a Linux container with hard CPU and
memory limits set *below* a Raspberry Pi 5 (2 CPUs / 1 GB, vs the Pi 5's 4 cores
and >=2 GB), enforced by cgroups, and reports per-decision latency + peak memory.

## Reproduce
From the repo root (after `run_all` has produced the models under `models/`):

```bash
cp models/lstm_ae_better_intersection.keras \
   models/scaler_better_intersection.pkl \
   models/rf_better_intersection.pkl \
   experiments/container_bench/
cp data/scenarios/better_intersection/eval/timing_manipulation_6000.csv \
   experiments/container_bench/scenario.csv
cd experiments/container_bench
docker build --platform linux/arm64 -t guardbox-bench .
docker run --rm --platform linux/arm64 --cpus=2 --memory=1g guardbox-bench
```

## Measured result (2 CPUs, 1 GB enforced)
- cgroup memory.max = 1.07 GB, cpu.max = 2.00 CPUs
- mean decision time: 42.5 ms  (LSTM reconstruction + RF classification)
- peak resident memory: 0.65 GB
- headroom vs the 5 s snapshot interval: 118x
