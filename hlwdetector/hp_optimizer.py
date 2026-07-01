"""Hyperparameter optimizer using Optuna"""

import optuna
from pathlib import Path

from hlwdetector.config.hpo_config import HPOConfig

class HPOptimizer:
    def __init__(self, hpo_config: str | Path):
        self.config = None

    def _generate_experiment_configs():
        pass

    def _objective(trial):
        pass

    def run_study():
        pass