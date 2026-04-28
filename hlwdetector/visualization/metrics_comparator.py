"""MetricsComparator — load and compare metrics across experiment directories."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from hlwdetector.adapters.base import MetricsDict

logger = logging.getLogger(__name__)

# For each output column, JSON keys to try in order:
#   tracker format → MetricsDict format → YOLO-native format
_KEY_MAP: dict[str, list[str]] = {
    "accuracy":   ["val/accuracy", "accuracy"],
    "precision":  ["val/precision", "precision", "metrics/precision(B)"],
    "recall":     ["val/recall",    "recall",    "metrics/recall(B)"],
    "f1":         ["val/f1",        "f1"],
    "mAP50":      ["val/mAP50",     "map50",     "metrics/mAP50(B)"],
    "mAP50:0.95": ["val/mAP50_95",  "map50_95",  "metrics/mAP50-95(B)"],
}

_COLUMNS = list(_KEY_MAP.keys())


def _lookup(d: dict, keys: list[str]) -> float | None:
    for k in keys:
        if k in d:
            v = d[k]
            return float(v) if v is not None else None
    return None


class MetricsComparator:
    """Aggregate and compare evaluation metrics across multiple experiments.

    Rows correspond to experiments; columns are accuracy, precision, recall,
    f1, mAP50, and mAP50:0.95.
    """

    def __init__(self, rows: list[dict], labels: list[str]) -> None:
        self._rows = rows
        self._labels = labels

    @classmethod
    def from_experiment_dirs(
        cls,
        dirs: list[str | Path],
        labels: list[str] | None = None,
    ) -> "MetricsComparator":
        """Load metrics from a list of experiment output directories.

        Args:
            dirs: Paths to experiment output directories (each containing metrics.json).
            labels: Row labels for the table. Defaults to config_name from config.json,
                    or the directory name if config.json is absent.
        """
        rows: list[dict] = []
        resolved_labels: list[str] = []

        for i, d in enumerate(dirs):
            d = Path(d)
            metrics_path = d / "metrics.json"
            config_path = d / "config.json"

            label = labels[i] if (labels and i < len(labels)) else None
            if label is None:
                if config_path.exists():
                    try:
                        label = json.loads(config_path.read_text()).get("config_name", d.name)
                    except Exception:
                        label = d.name
                else:
                    label = d.name

            if not metrics_path.exists():
                logger.warning("metrics.json not found in %s — row will be all NaN.", d)
                rows.append({col: None for col in _COLUMNS})
                resolved_labels.append(label)
                continue

            raw = json.loads(metrics_path.read_text())
            rows.append({col: _lookup(raw, _KEY_MAP[col]) for col in _COLUMNS})
            resolved_labels.append(label)

        return cls(rows, resolved_labels)

    @classmethod
    def from_metrics_dicts(
        cls,
        metrics_dicts: list["MetricsDict"],
        labels: list[str],
    ) -> "MetricsComparator":
        """Build a comparator from in-memory MetricsDict objects.

        Args:
            metrics_dicts: List of MetricsDict returned by adapter.evaluate().
            labels: Experiment label for each entry.
        """
        rows = [
            {
                "accuracy":   m.accuracy,
                "precision":  m.precision,
                "recall":     m.recall,
                "f1":         m.f1,
                "mAP50":      m.map50,
                "mAP50:0.95": m.map50_95,
            }
            for m in metrics_dicts
        ]
        return cls(rows, list(labels))

    def to_dataframe(self) -> pd.DataFrame:
        """Return a DataFrame with experiments as index and metrics as columns."""
        df = pd.DataFrame(self._rows, index=self._labels, columns=_COLUMNS)
        df.index.name = "experiment"
        return df

    def to_csv(self, path: str | Path) -> None:
        """Save the comparison table to a CSV file."""
        path = Path(path)
        self.to_dataframe().to_csv(path)
        logger.info("Metrics comparison saved to %s", path)

    def __repr__(self) -> str:
        return self.to_dataframe().to_string()

    def __str__(self) -> str:
        return self.to_dataframe().to_string()
