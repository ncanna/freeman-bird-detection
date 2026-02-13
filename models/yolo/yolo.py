from ultralytics import YOLO
from ultralytics import settings
from ultralytics.data.split import autosplit
import os
from pathlib import Path


class YOLODetector:
    def __init__(self, model_name="yolo11n.pt"):
        self.model = YOLO(model_name)
        script_dir = Path(__file__).parent
        runs_dir = script_dir / ".." / "outputs" / f"{model_name}" / "runs"
        runs_dir = runs_dir.resolve()
        settings.update({"runs_dir": runs_dir, "tensorboard": True})

    def train(self, data_path, epochs=100):
        self.model.train(data=data_path, epochs=epochs)

    def detect(self, test_data_path):
        results = self.model(test_data_path)
        return results