"""Run one Optuna HPO study from a YAML configuration."""

from __future__ import annotations

import argparse

from hlwdetector.hp_optimizer import HPOptimizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to an HPO YAML config")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    optimizer = HPOptimizer(args.config)
    study = optimizer.run_study()
    print(
        f"Best trial: {study.best_trial.number}; "
        f"value={study.best_trial.value}; params={study.best_trial.params}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
