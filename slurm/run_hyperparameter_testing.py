#!/usr/bin/env python3
"""Prepare and optionally submit the H23 hyperparameter sensitivity study.

The default invocation is a safe preview: it prints every planned ``sbatch``
command without writing configs or submitting jobs. Use ``--write-configs`` to
inspect generated YAML files, or ``--submit`` to write them and submit one SLURM
job per run through the existing ``slurm/run_experiment.py`` workflow.

Examples:
    python slurm/run_hyperparameter_testing.py --date 20260714
    python slurm/run_hyperparameter_testing.py --write-configs
    python slurm/run_hyperparameter_testing.py --submit
    python slurm/run_hyperparameter_testing.py --models rtdetr --parameters learning_rate
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from hlwdetector.config import ExperimentConfig


EPOCHS = 50
WANDB_PROJECT = "bird-detection"
WANDB_ENTITY = "gatech-birdlab"
STUDY_NAME = "h23_hyperparameter_sensitivity"

MODEL_ORDER = ("yolo11", "yolo26", "rtdetr")
BASE_CONFIGS = {
    "yolo11": REPO_ROOT / "configs/yolo11_h23_full.yaml",
    "yolo26": REPO_ROOT / "configs/yolo26_h23_full.yaml",
    "rtdetr": REPO_ROOT / "configs/rtdetr_h23_full.yaml",
}
EXPECTED_BASELINE_HYPERPARAMETERS = {
    "yolo11": {
        "model_weights": "yolo11n.pt",
        "epochs": EPOCHS,
        "imgsz": 640,
        "batch": 32,
        "amp": True,
        "optimizer": "MuSGD",
        "lr0": 0.01,
        "momentum": 0.9,
        "warmup_bias_lr": 0.0,
        "weight_decay": 0.0005,
        "mosaic": 1.0,
    },
    "yolo26": {
        "model_weights": "yolo26n.pt",
        "epochs": EPOCHS,
        "imgsz": 640,
        "batch": 32,
        "amp": True,
        "optimizer": "MuSGD",
        "lr0": 0.01,
        "momentum": 0.9,
        "warmup_bias_lr": 0.0,
        "weight_decay": 0.0005,
        "mosaic": 1.0,
    },
    "rtdetr": {
        "model_weights": "rtdetr-l.pt",
        "epochs": EPOCHS,
        "imgsz": 640,
        "batch": 32,
        "amp": True,
        "optimizer": "AdamW",
        "lr0": 0.0001,
        "momentum": 0.9,
        "warmup_bias_lr": 0.0,
        "weight_decay": 0.0005,
        "mosaic": 1.0,
    },
}

# Values listed here are the non-default OFAT runs. Defaults are read from the
# full H23 configs and represented once by each model's shared baseline run.
SWEEP_ORDER = ("learning_rate", "weight_decay", "mosaic")
SWEEPS: dict[str, dict[str, Any]] = {
    "learning_rate": {
        "config_key": "lr0",
        "values": {
            "yolo11": (("low", 0.001), ("high", 0.02)),
            "yolo26": (("low", 0.001), ("high", 0.02)),
            "rtdetr": (("low", 0.00001), ("high", 0.001)),
        },
    },
    "weight_decay": {
        "config_key": "weight_decay",
        "values": {
            "yolo11": (("low", 0.0001), ("high", 0.001)),
            "yolo26": (("low", 0.0001), ("high", 0.001)),
            "rtdetr": (("low", 0.0001), ("high", 0.001)),
        },
    },
    "mosaic": {
        "config_key": "mosaic",
        "values": {
            # mosaic=1.0 is the default/high candidate and is already covered
            # by the baseline, so only the non-default low run is submitted.
            "yolo11": (("low", 0.0),),
            "yolo26": (("low", 0.0),),
            "rtdetr": (("low", 0.0),),
        },
    },
}


@dataclass(frozen=True)
class RunSpec:
    sequence: int
    run_date: str
    model: str
    parameter: str
    config_key: str | None
    setting_label: str
    value: Any
    default_value: Any

    @property
    def is_baseline(self) -> bool:
        return self.config_key is None

    @property
    def run_name(self) -> str:
        number = f"{self.sequence:03d}"
        if self.is_baseline:
            return f"baseline_{self.model}_default_h23_{self.run_date}_{number}"
        return (
            f"ab_{self.model}_{self.parameter}_{self.setting_label}_"
            f"h23_{self.run_date}_{number}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date",
        default=datetime.now().strftime("%Y%m%d"),
        help="Run date used in names (YYYYMMDD; default: today).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_ORDER,
        default=list(MODEL_ORDER),
        help="Models to include; numbering remains stable relative to the full matrix.",
    )
    parser.add_argument(
        "--parameters",
        nargs="+",
        choices=SWEEP_ORDER,
        default=list(SWEEP_ORDER),
        help="Sensitivity parameters to include.",
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Omit the three shared default baseline runs.",
    )
    parser.add_argument(
        "--write-configs",
        action="store_true",
        help="Write generated YAML configs but do not submit jobs.",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Write configs and submit jobs. Without this flag no jobs are submitted.",
    )
    parser.add_argument(
        "--generated-dir",
        type=Path,
        default=REPO_ROOT / "slurm/generated/hyperparameter_testing",
        help="Persistent directory for generated configs and the submission manifest.",
    )
    parser.add_argument(
        "--sbatch-script",
        type=Path,
        default=REPO_ROOT / "slurm/train_yolo.sbatch",
        help="Existing sbatch wrapper to use for every model (default: H200 wrapper).",
    )
    return parser.parse_args()


def validate_date(value: str) -> str:
    if not re.fullmatch(r"\d{8}", value):
        raise ValueError(f"--date must use YYYYMMDD, got {value!r}")
    datetime.strptime(value, "%Y%m%d")
    return value


def load_base_configs() -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}
    for model in MODEL_ORDER:
        path = BASE_CONFIGS[model]
        config = ExperimentConfig.from_yaml(str(path))
        raw = dataclasses.asdict(config)
        hyperparameters = raw["hyperparameters"]

        expected = EXPECTED_BASELINE_HYPERPARAMETERS[model]
        mismatches = {
            key: {"expected": value, "found": hyperparameters.get(key)}
            for key, value in expected.items()
            if hyperparameters.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"{path.relative_to(REPO_ROOT)} does not match the documented "
                f"{model} baseline: {mismatches}"
            )
        for sweep_name in SWEEP_ORDER:
            key = SWEEPS[sweep_name]["config_key"]
            if key not in hyperparameters:
                raise ValueError(f"{path.relative_to(REPO_ROOT)} is missing default {key!r}")

        configs[model] = raw
    return configs


def build_run_specs(run_date: str, base_configs: dict[str, dict[str, Any]]) -> list[RunSpec]:
    specs: list[RunSpec] = []
    sequence = 1

    for model in MODEL_ORDER:
        specs.append(
            RunSpec(
                sequence=sequence,
                run_date=run_date,
                model=model,
                parameter="baseline",
                config_key=None,
                setting_label="default",
                value=None,
                default_value=None,
            )
        )
        sequence += 1

    for parameter in SWEEP_ORDER:
        sweep = SWEEPS[parameter]
        config_key = sweep["config_key"]
        for model in MODEL_ORDER:
            default_value = base_configs[model]["hyperparameters"][config_key]
            for setting_label, value in sweep["values"][model]:
                if value == default_value:
                    # A shared baseline already represents every default value.
                    continue
                specs.append(
                    RunSpec(
                        sequence=sequence,
                        run_date=run_date,
                        model=model,
                        parameter=parameter,
                        config_key=config_key,
                        setting_label=setting_label,
                        value=value,
                        default_value=default_value,
                    )
                )
                sequence += 1

    names = [spec.run_name for spec in specs]
    if len(names) != len(set(names)):
        raise RuntimeError("Generated duplicate run names; check the model/sweep definitions")
    return specs


def make_run_config(spec: RunSpec, base_config: dict[str, Any]) -> dict[str, Any]:
    run_config = copy.deepcopy(base_config)
    baseline_hyperparameters = base_config["hyperparameters"]
    hyperparameters = run_config["hyperparameters"]

    if spec.config_key is not None:
        hyperparameters[spec.config_key] = spec.value

    changed = {
        key
        for key in set(baseline_hyperparameters) | set(hyperparameters)
        if baseline_hyperparameters.get(key) != hyperparameters.get(key)
    }
    expected = set() if spec.is_baseline else {spec.config_key}
    if changed != expected:
        raise RuntimeError(
            f"{spec.run_name} changes {sorted(changed)}; expected only {sorted(expected)}"
        )
    if hyperparameters["epochs"] != EPOCHS:
        raise RuntimeError(f"{spec.run_name} does not use exactly {EPOCHS} epochs")

    if spec.is_baseline:
        comparison_roles = [
            {
                "hyperparameter": "lr0",
                "display_name": "learning_rate",
                "setting_label": "default",
                "value": baseline_hyperparameters["lr0"],
            },
            {
                "hyperparameter": "weight_decay",
                "display_name": "weight_decay",
                "setting_label": "default",
                "value": baseline_hyperparameters["weight_decay"],
            },
            {
                "hyperparameter": "mosaic",
                "display_name": "mosaic",
                "setting_label": "default_high",
                "value": baseline_hyperparameters["mosaic"],
            },
        ]
    else:
        comparison_roles = [
            {
                "hyperparameter": spec.config_key,
                "display_name": spec.parameter,
                "setting_label": spec.setting_label,
                "value": spec.value,
            }
        ]
    comparison_tags = [
        f"comparison-{role['display_name']}-{role['setting_label'].replace('_', '-')}"
        for role in comparison_roles
    ]

    run_config.update(
        {
            "run_name": spec.run_name,
            "wandb_project": WANDB_PROJECT,
            "wandb_entity": WANDB_ENTITY,
            "wandb_group": f"{STUDY_NAME}_{spec.run_date}",
            "wandb_tags": [
                "h23",
                "hyperparameter-sensitivity",
                spec.model,
                spec.parameter,
                spec.setting_label,
                *comparison_tags,
            ],
            "resume_experiment": None,
            "resume_from": None,
            "experiment_metadata": {
                "study": STUDY_NAME,
                "dataset": "h23",
                "model": spec.model,
                "model_weights": hyperparameters["model_weights"],
                "tested_hyperparameter": spec.config_key or "baseline",
                "hyperparameter_display_name": spec.parameter,
                "setting_label": spec.setting_label,
                "tested_value": spec.value,
                "default_value": spec.default_value,
                "default_or_modified": "default" if spec.is_baseline else "modified",
                "is_baseline": spec.is_baseline,
                "epochs": EPOCHS,
                "run_date": spec.run_date,
                "run_number": f"{spec.sequence:03d}",
                "slurm_job_name": spec.run_name,
                "comparison_roles": comparison_roles,
                "baseline_hyperparameters": {
                    name: baseline_hyperparameters[SWEEPS[name]["config_key"]]
                    for name in SWEEP_ORDER
                },
            },
        }
    )
    return run_config


def resolve_cli_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def config_path_for(spec: RunSpec, generated_root: Path) -> Path:
    return generated_root / spec.run_date / "configs" / f"{spec.run_name}.yaml"


def write_config(path: Path, config: dict[str, Any]) -> None:
    contents = yaml.safe_dump(config, sort_keys=False)
    if path.exists():
        if path.read_text(encoding="utf-8") != contents:
            raise FileExistsError(
                f"Refusing to change existing generated config {path}. "
                "Use a new --date or remove the unsubmitted file explicitly."
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def sbatch_command(spec: RunSpec, config_path: Path, sbatch_script: Path) -> list[str]:
    return [
        "sbatch",
        f"--job-name={spec.run_name}",
        f"--chdir={REPO_ROOT}",
        str(sbatch_script),
        str(config_path),
    ]


def load_manifest(path: Path, run_date: str) -> dict[str, Any]:
    if not path.exists():
        return {"study": STUDY_NAME, "run_date": run_date, "submissions": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("study") != STUDY_NAME or data.get("run_date") != run_date:
        raise ValueError(f"Unexpected submission manifest contents: {path}")
    data.setdefault("submissions", {})
    return data


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def submit_job(command: list[str]) -> tuple[str, str]:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    message = result.stdout.strip()
    match = re.search(r"Submitted batch job (\d+)", message)
    if not match:
        raise RuntimeError(f"Could not parse SLURM job ID from: {message!r}")
    return match.group(1), message


def main() -> int:
    args = parse_args()
    try:
        run_date = validate_date(args.date)
        base_configs = load_base_configs()
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    selected_models = set(args.models)
    selected_parameters = set(args.parameters)
    all_specs = build_run_specs(run_date, base_configs)
    specs = [
        spec
        for spec in all_specs
        if spec.model in selected_models
        and (
            (spec.is_baseline and not args.skip_baselines)
            or (not spec.is_baseline and spec.parameter in selected_parameters)
        )
    ]

    generated_root = resolve_cli_path(args.generated_dir)
    sbatch_script = resolve_cli_path(args.sbatch_script)
    if not sbatch_script.exists():
        print(f"SBATCH script not found: {sbatch_script}", file=sys.stderr)
        return 2
    if args.submit and shutil.which("sbatch") is None:
        print("sbatch is not available on PATH; no jobs were submitted", file=sys.stderr)
        return 2
    if args.submit:
        # The existing sbatch wrapper writes to this ignored directory, and
        # SLURM requires its parent to exist before the job starts.
        (REPO_ROOT / "slurm/logs").mkdir(parents=True, exist_ok=True)

    write_configs = args.write_configs or args.submit
    manifest_path = generated_root / run_date / "submissions.json"
    manifest = load_manifest(manifest_path, run_date) if args.submit else None
    submitted = 0
    skipped = 0

    print(f"Prepared {len(specs)} run(s) for {WANDB_ENTITY}/{WANDB_PROJECT}:")
    for spec in specs:
        config = make_run_config(spec, base_configs[spec.model])
        config_path = config_path_for(spec, generated_root)
        command = sbatch_command(spec, config_path, sbatch_script)
        print(f"{spec.sequence:03d}  {spec.run_name}")
        print(f"     {shlex.join(command)}")

        if not args.submit:
            if write_configs:
                try:
                    write_config(config_path, config)
                except FileExistsError as exc:
                    print(f"Config generation failed: {exc}", file=sys.stderr)
                    return 2
            continue

        assert manifest is not None
        existing = manifest["submissions"].get(spec.run_name)
        output_dir = Path(config["output_dir"]) / spec.run_name
        if existing is not None:
            print(f"     SKIP: already recorded as SLURM job {existing['job_id']}")
            skipped += 1
            continue
        if output_dir.exists():
            print(f"     SKIP: output directory already exists: {output_dir}")
            skipped += 1
            continue

        # Write only after duplicate checks. A queued job may not have opened
        # its YAML yet, so already-submitted config files must remain immutable.
        try:
            write_config(config_path, config)
        except FileExistsError as exc:
            print(f"Config generation failed: {exc}", file=sys.stderr)
            return 2

        try:
            job_id, message = submit_job(command)
        except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
            print(f"Submission failed for {spec.run_name}: {exc}", file=sys.stderr)
            return 1
        print(f"     {message}")
        manifest["submissions"][spec.run_name] = {
            "job_id": job_id,
            "config": str(config_path),
            "submitted_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        save_manifest(manifest_path, manifest)
        submitted += 1

    if args.submit:
        print(f"Submitted {submitted} job(s); skipped {skipped} duplicate/existing run(s).")
        print(f"Submission manifest: {manifest_path}")
    elif args.write_configs:
        print(f"Wrote {len(specs)} config(s) under {generated_root / run_date}; no jobs submitted.")
    else:
        print("Preview only: no configs were written and no SLURM jobs were submitted.")
        print("Re-run with --write-configs to inspect YAML or --submit to submit jobs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
