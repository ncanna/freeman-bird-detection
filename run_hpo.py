"""Run an Optuna hyperparameter study from the command line."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

DEFAULT_CONFIGS = {
    "yolo": "configs/hpo/yolo26_hpo.yaml",
    "rtdetr": "configs/hpo/rtdetr_hpo.yaml",
    "swin": "configs/hpo/swin_hpo.yaml",
    "detr": "configs/hpo/detr_hpo.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run model hyperparameter optimization (HPO) with Optuna."
    )
    parser.add_argument(
        "--model",
        choices=DEFAULT_CONFIGS,
        default=os.environ.get("HPO_MODEL", "yolo"),
        help="model adapter to optimize (default: yolo)",
    )
    parser.add_argument(
        "--config",
        default=os.environ.get("HPO_CONFIG") or None,
        help="HPO YAML config; defaults to the matching configs/hpo file",
    )
    parser.add_argument(
        "--coco-json",
        default=os.environ.get("COCO_JSON") or None,
        help="override the COCO annotations path in the config",
    )
    parser.add_argument(
        "--split-json",
        default=os.environ.get("SPLIT_JSON") or None,
        help="override the video-stem train/val/test split JSON path",
    )
    parser.add_argument(
        "--images-dir",
        default=os.environ.get("IMAGES_DIR") or None,
        help="override the directory containing extracted frame images",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate the selected config and dataset paths without training",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config or DEFAULT_CONFIGS[args.model])

    # Delay ML imports so --help works even outside the PACE environment.
    from hlwdetector.config.hpo_config import HPOConfig
    import hlwdetector.adapters  # noqa: F401 - register model adapters for validation

    config = HPOConfig.from_yaml(str(config_path))
    if config.model_name != args.model:
        raise SystemExit(
            f"Config {config_path} selects model {config.model_name!r}, "
            f"but --model selected {args.model!r}. Choose a matching config/model."
        )

    for argument, field in (
        (args.coco_json, "coco_json"),
        (args.split_json, "split_json"),
        (args.images_dir, "images_dir"),
    ):
        if argument:
            setattr(config, field, str(Path(argument).expanduser().resolve()))

    config.validate()
    print(f"Model:       {config.model_name}")
    print(f"Study:       {config.study_args.study_name}")
    print(f"Trials:      {config.optimize_args.n_trials}")
    print(f"COCO JSON:   {config.coco_json}")
    print(f"Split JSON:  {config.split_json}")
    print(f"Images dir:  {config.images_dir}")

    if args.validate_only:
        print("Configuration and dataset paths are valid; no training was started.")
        return

    from hlwdetector.hp_optimizer import HPOptimizer

    HPOptimizer(config).run_study()


if __name__ == "__main__":
    main()