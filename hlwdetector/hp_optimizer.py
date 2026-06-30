"""Hyperparameter optimizer using Optuna"""

import optuna

class HPOptimizer:
    def __init__(self, model_name, hparam_ranges, wandb_project="bird-detection"):
        self.model_name = model_name
        self.hparam_ranges = hparam_ranges
        self.wandb_project = wandb_project

    def _generate_configs():
        pass

    def _objective():
        pass

    def run_study():
        pass