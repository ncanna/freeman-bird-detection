"""CLI for a System One frame experiment.

    python -m system_one_detector.main system_one_detector/configs/jev_omni_h23.yaml
    python -m system_one_detector.main system_one_detector/configs/laya_h23.yaml --dry-run --limit 4
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _bootstrap() -> None:
    if __package__:
        return
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_bootstrap()

from system_one_detector.engine import run_experiment  # noqa: E402
from system_one_detector.models.registry import get_backend  # noqa: E402
from system_one_detector.util.config import KNOWN_BACKENDS, SystemOneConfig  # noqa: E402


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m system_one_detector.main",
        description="Ask a System One model typed questions about camera-trap frames.",
    )
    parser.add_argument("config", nargs="?", default=None, help="YAML config path. Paths inside it are relative to the file.")
    parser.add_argument("--limit", type=int, default=None, help="Override the config frame limit.")
    parser.add_argument("--split", choices=("train", "val", "test"), default=None)
    parser.add_argument("--backend", default=None, help=f"Override backend. One of: {', '.join(sorted(KNOWN_BACKENDS))}.")
    parser.add_argument("--dry-run", action="store_true", help="Walk the split with the mock backend. No API call, no weights.")
    parser.add_argument("--list-backends", action="store_true", help="Print registered backends and whether each sees pixels.")
    parser.add_argument("--output-dir", default=None)
    return parser


def _backend_summary() -> list[str]:
    from system_one_detector.models import list_backends
    from system_one_detector.models.base import DecisionBackend

    rows = []
    for name in list_backends():
        cls = get_backend(name)
        pixels = "sees pixels" if cls.sees_pixels else "text only"
        rows.append(f"{name:<18} {pixels:<12} {cls.__module__}")
    return rows


def main(argv: list[str] | None = None) -> Path | None:
    args = get_args_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.list_backends:
        for backend in _backend_summary():
            print(backend)
        return None
    if args.config is None:
        get_args_parser().error("config is required unless --list-backends is given")
    config = SystemOneConfig.from_yaml(args.config)
    if args.limit is not None:
        config.limit = args.limit
    if args.split is not None:
        config.split = args.split
    if args.backend is not None:
        config.backend = args.backend
    if args.output_dir is not None:
        config.output_dir = str(Path(args.output_dir).resolve())
    return run_experiment(config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
