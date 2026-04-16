"""Model adapter registry — import both adapters to trigger @register_adapter decoration."""

from hlwdetector.adapters import yolo_adapter  # noqa: F401
from hlwdetector.adapters import megadetector_adapter  # noqa: F401
from hlwdetector.adapters import rtdetr_adapter  # noqa: F401
