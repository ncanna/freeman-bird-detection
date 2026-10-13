"""Run identical frame/question evaluations against several model IDs.

Example:
    python -m system_one_detector.sweep \
        system_one_detector/configs/h55_ollama_clef.yaml \
        --models clef clef-flash --limit 20
"""

from __future__ import annotations

import argparse
import logging
import re

from system_one_detector.engine import run_experiment
from system_one_detector.util.config import SystemOneConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="base YAML config shared by all models")
    parser.add_argument("--models", nargs="+", required=True, help="model IDs understood by the configured backend")
    parser.add_argument("--limit", type=int, help="evaluate the same first N frames for every model")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    failed = []
    for model in args.models:
        config = SystemOneConfig.from_yaml(args.config)
        config.model = model
        suffix = re.sub(r"[^A-Za-z0-9_.-]+", "_", model).strip("_.-") or "model"
        config.config_name = f"{config.config_name}_{suffix}"
        if args.limit is not None:
            config.limit = args.limit
        try:
            output = run_experiment(config)
            print(f"{model}: {output}")
        except Exception as exc:  # preserve later model runs when one backend fails
            failed.append((model, exc))
            logging.exception("model run failed: %s", model)
    if failed:
        print("Failed model runs:")
        for model, exc in failed:
            print(f"  {model}: {type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
