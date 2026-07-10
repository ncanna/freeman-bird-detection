"""Hyperparameter optimizer using Optuna"""

import optuna
from pathlib import Path
from typing import Dict

from hlwdetector.config.hpo_config import HPOConfig
from hlwdetector.config.experiment_config import ExperimentConfig
from hlwdetector.runner import ExperimentRunner 

class HPOptimizer:
    def __init__(self, hpo_config: str | Path):
        # Load HPO config from YAML
        self.config = HPOConfig.from_yaml(hpo_config)

    def _generate_experiment_config(self, sampled_hparams: Dict, trial_number: int):
        """
        Dynamically generate experiment configs by combinging sampled hyperparameters with the hpo config at runtime.
        """
        cfg = self.config
        static_hparams = cfg.hyperparameters.get("static")
        print("********************************************************")
        print(f"STATIC HPARAMS: {static_hparams}")
        print("********************************************************")
        hparams = {**static_hparams, **sampled_hparams}

        return ExperimentConfig(
            model_name=cfg.model_name,
            config_name=f"{cfg.study_name}_trial_{trial_number}",
            hyperparameters=hparams,
            coco_json=cfg.coco_json,
            images_dir=cfg.images_dir,
            split_json=cfg.split_json,
            output_dir=cfg.output_dir,
            random_seed=cfg.random_seed,
            wandb_project=cfg.wandb_project,
        )

    def _extract_hparam_search_space(self, spec):
        """
        Return the low and high values of the search space for each hyperparameter of type int
        or float in addition to any keyword args specified in the config.
        """
        low = spec[0]
        high = spec[1]
        kwargs = {}
        for arg in spec[2:]:
            kwargs.update(arg)
        
        return low, high, kwargs

    def _objective(self, trial):
        """
        maximize the configured metric by dynamically creating experiment runners from the
        HPO config and sampled hyperparameters at runtime.
        """
        suggest_fns = {
            "categorical": trial.suggest_categorical,
            "int": trial.suggest_int,
            "float": trial.suggest_float,
        }
        # Sample hyperparameter search space
        hparams = self.config.hyperparameters
        sampled_hparams = {}
        for category, suggest in suggest_fns.items():
            search_space = hparams[category]
            if not search_space:  # If no hyperparameters are configured for a given category then skip
                continue
            for hp, spec in search_space.items():
                low, high, kwargs = self._extract_hparam_search_space(spec)
                sampled_hparams[hp] = suggest(hp, low, high, **kwargs)

        # Generate experiment conifg and train model
        experiment_config = self._generate_experiment_config(sampled_hparams, trial.number)
        print("********************************************************")
        print(f"EXPERIMENT CONFIG: {experiment_config}")
        print("********************************************************")
        runner = ExperimentRunner(experiment_config)
        runner.train()
        metrics = runner.evaluate()
        target_metric = self.config.metric

        return metrics[target_metric]

    def run_study(self):
        """
        Run Optuna study >><))))*>
        """
        study = optuna.create_study()
        #if self.config.n_trials is not None: # Remove conditional statements and instead construct kwargs for study parameters
        #    study.optimize(self._objective, n_trials=self.config.n_trials)
        study.optimize(self._objective, **self.config.optuna)
        #else:
        #    study.optimize(self._objective)
