"""Model adapter registry — import both adapters to trigger @register_adapter decoration."""

from experiments.adapters import yolo_adapter  # noqa: F401
from experiments.adapters import megadetector_adapter  # noqa: F401
