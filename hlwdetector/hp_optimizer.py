"""Hyperparameter optimizer using Optuna"""

import optuna
from datetime import datetime
from pathlib import Path
from typing import Dict
from dataclasses import asdict

from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from hlwdetector.config.hpo_config import HPOConfig
from hlwdetector.config.experiment_config import ExperimentConfig
from hlwdetector.runner import ExperimentRunner

# Maps MetricsDict field names (used by config `metric`) to the Ultralytics per-epoch
# metric keys reported in the on_fit_epoch_end callback. "f1" has no direct per-epoch
# key and is derived from precision/recall in _epoch_metric_value().
_METRIC_KEY_MAP = {
    "precision": "metrics/precision(B)",
    "recall":    "metrics/recall(B)",
    "map50":     "metrics/mAP50(B)",
    "map50_95":  "metrics/mAP50-95(B)",
}


def _epoch_metric_value(metric: str, metrics: dict):
    """Extract `metric` (a MetricsDict field name) from an Ultralytics per-epoch metrics
    dict, deriving f1 from precision/recall. Returns None if the value is unavailable."""
    if metric == "f1":
        p = metrics.get(_METRIC_KEY_MAP["precision"])
        r = metrics.get(_METRIC_KEY_MAP["recall"])
        if p is None or r is None or (p + r) == 0:
            return None
        return 2 * p * r / (p + r)
    key = _METRIC_KEY_MAP.get(metric)
    if key is None:
        return None
    return metrics.get(key)


class HPOptimizer:
    def __init__(self, hpo_config: str | Path):
        # Load HPO config from YAML
        self.config = HPOConfig.from_yaml(hpo_config)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.id = f"{self.config.study_args.study_name}_{self.timestamp}"

    def _generate_experiment_config(self, sampled_hparams: Dict, trial_number: int):
        """
        Dynamically generate experiment configs by combinging sampled hyperparameters with the hpo config at runtime.
        """
        cfg = self.config
        static_hparams = cfg.hyperparameters.get("static")
        hparams = {**static_hparams, **sampled_hparams}

        return ExperimentConfig(
            model_name=cfg.model_name,
            config_name=f"{cfg.study_args.study_name}_trial_{trial_number}",
            hyperparameters=hparams,
            coco_json=cfg.coco_json,
            images_dir=cfg.images_dir,
            split_json=cfg.split_json,
            output_dir=cfg.output_dir,
            random_seed=cfg.random_seed,
            wandb_project=cfg.wandb_project,
            wandb_group=self.id,
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

    def _build_sampler(self, name: str | None):
        """Translate a sampler name from the config into an Optuna sampler instance."""
        if name is None:
            return None
        samplers = {
            "tpe": optuna.samplers.TPESampler,
            "random": optuna.samplers.RandomSampler,
            "grid": optuna.samplers.GridSampler,
            "cmaes": optuna.samplers.CmaEsSampler,
        }
        key = name.lower()
        if key not in samplers:
            raise ValueError(
                f"Unknown sampler {name!r}; expected one of {sorted(samplers)}"
            )
        return samplers[key](seed=self.config.random_seed)

    def _build_pruner(self, name: str | None):
        """Translate a pruner name from the config into an Optuna pruner instance."""
        if name is None or name.lower() == "none":
            return optuna.pruners.NopPruner()
        pruners = {
            "median": optuna.pruners.MedianPruner,
            "hyperband": optuna.pruners.HyperbandPruner,
            "successivehalving": optuna.pruners.SuccessiveHalvingPruner,
        }
        key = name.lower()
        if key not in pruners:
            raise ValueError(
                f"Unknown pruner {name!r}; expected one of {sorted(pruners)}"
            )
        return pruners[key]()
    
    def _build_storage(self, storage: str | None):
        """Resolve the Optuna storage backend for the study.

        If the config specifies an explicit storage URL, pass it through
        unchanged. Otherwise default to a Journal file backend rooted under the
        config output directory, one journal file per study run:
        ``{output_dir}/hpo/{study_name}_{timestamp}``.
        """
        if storage is not None:
            return storage
        
        journal_path = (
            Path(self.config.output_dir) / "hpo" / f"{self.id}"
        )
        journal_path.parent.mkdir(parents=True, exist_ok=True)
        return JournalStorage(JournalFileBackend(str(journal_path)))

    def _objective(self, trial):
        """
        Optimize the configured metric by dynamically creating experiment runners from the
        HPO config and sampled hyperparameters at runtime.
        """
        # Sample hyperparameter search space
        hparams = self.config.hyperparameters
        sampled_hparams = {}

        # Categorical: the whole spec is the list of choices, passed to
        # trial.suggest_categorical(name, choices) as a single argument.
        for hp, choices in (hparams.get("categorical") or {}).items():
            sampled_hparams[hp] = trial.suggest_categorical(hp, choices)

        # Int / float: spec is [low, high, {**kwargs}].
        range_suggest_fns = {
            "int": trial.suggest_int,
            "float": trial.suggest_float,
        }
        for category, suggest in range_suggest_fns.items():
            search_space = hparams.get(category)
            if not search_space:  # If no hyperparameters are configured for a given category then skip
                continue
            for hp, spec in search_space.items():
                low, high, kwargs = self._extract_hparam_search_space(spec)
                sampled_hparams[hp] = suggest(hp, low, high, **kwargs)

        # Generate experiment conifg and train model
        experiment_config = self._generate_experiment_config(sampled_hparams, trial.number)
        runner = ExperimentRunner(experiment_config)

        # Report per-epoch metrics to Optuna so the pruner can stop losing trials early.
        def _pruning_callback(epoch, metrics):
            value = _epoch_metric_value(self.config.metric, metrics)
            if value is None:
                return
            trial.report(value, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        runner.adapter._hpo_pruning_callback = _pruning_callback
        runner.train()
        metrics = runner.evaluate()
        runner.tracker.finish()
        target_metric = self.config.metric

        return getattr(metrics, target_metric)

    def run_study(self):
        """
        Run Optuna study  ><)))°>  <°(((><
        """
        study_kwargs = asdict(self.config.study_args)        # kwargs for optuna.create_study
        optimize_kwargs = asdict(self.config.optimize_args)  # kwargs for optuna.study.optimize
        # Optuna wants sampler/pruner objects, not the name strings stored in the config.
        study_kwargs["sampler"] = self._build_sampler(study_kwargs.get("sampler"))
        study_kwargs["pruner"] = self._build_pruner(study_kwargs.get("pruner"))
        study_kwargs["storage"] = self._build_storage(study_kwargs.get("storage"))
        study = optuna.create_study(**study_kwargs)
        study.optimize(self._objective, **optimize_kwargs)