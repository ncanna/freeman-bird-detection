"""Run a single experiment end-to-end (train -> eval -> predict -> visualize).

Usage:
    python slurm/run_experiment.py configs/experiment/yolo11_h23_full.yaml

Mirrors the run_experiments.ipynb pipeline but as a plain script so it can be
submitted to SLURM and survive SSH disconnects.
"""

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from hlwdetector.runner import ExperimentRunner

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python slurm/run_experiment.py <config_yaml>")
        sys.exit(1)

    config = sys.argv[1]
    start = time.time()
    ExperimentRunner(config).run_pipeline()
    print(f"[{config}] finished in {(time.time() - start) / 60:.2f} min")
