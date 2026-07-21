"""Run the full experiment suite in sequence (shared scenario cache).

  1. Core 375-scenario evaluation on the primary topology
  2. Baseline comparison
  3. Window-length study (12 vs 18)
  4. Runtime benchmark
  5. Multi-intersection generalization (all three topologies)
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from detection import config as C  # noqa: E402
from experiments import (run_evaluation, run_baselines, run_window_study,  # noqa: E402
                         run_runtime, run_multi_intersection)

PRIMARY = C.DEFAULT_TOPOLOGY
N_EVAL = C.SCENARIOS_PER_CLASS


def main():
    t0 = time.time()
    print("========== 1. CORE EVALUATION ==========", flush=True)
    run_evaluation.run(topo=PRIMARY, n_eval=N_EVAL, tag=PRIMARY)

    print("\n========== 2. BASELINES ==========", flush=True)
    run_baselines.run(topo=PRIMARY, n_eval=N_EVAL)

    print("\n========== 3. WINDOW STUDY ==========", flush=True)
    run_window_study.run(topo=PRIMARY, windows=(12, 18), n_eval=N_EVAL)

    print("\n========== 4. RUNTIME ==========", flush=True)
    run_runtime.run(topo=PRIMARY, n_eval=N_EVAL)

    print("\n========== 5. MULTI-INTERSECTION ==========", flush=True)
    run_multi_intersection.run(
        topos=("intersection", "better_intersection", "complex_intersection"),
        n_eval=25)

    print(f"\n[run_all] complete in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
