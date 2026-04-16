# Freeman Bird Detection

A framework for running and comparing bird detection experiments on camera trap footage. Supports multiple detection models through a unified adapter interface, with experiment tracking, artifact management, and visualization.

## Repository Structure

```
freeman-bird-detection/
├── hlwdetector/          # Core experiment framework
├── configs/              # YAML experiment configurations
├── data/                 # Datasets (h03, h23, african-wildlife)
├── outputs/              # Experiment artifacts and results
├── utilities/            # Data preparation and annotation conversion tools
├── run_experiments.py    # Example experiment runner script
└── run_experiments.ipynb # Interactive experiment notebook
```

## Prerequisites

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have the following inputs ready:
   - A COCO-format annotation JSON (e.g. `instances_merged.json`)
   - A split JSON mapping split names to video stems (see format below)
   - Extracted frame images in a single flat directory (`images_dir`)

If you need to extract frames from raw videos:
```python
from utilities.video_dataset_prep_tools import extract_frames_from_dir

extract_frames_from_dir(
    video_dir="/path/to/videos",
    output_dir="data/h23/images"
)
```

The `split.json` file maps split names to lists of video stems. The framework filters images at runtime by matching each frame's filename prefix against the listed stems:
```json
{
  "train": ["video_001", "video_002"],
  "val":   ["video_003"],
  "test":  ["video_004"]
}
```

---

## Defining a Config

Create a YAML file in `configs/`. All paths are resolved relative to the YAML file's location:

```yaml
config_name: yolo11_h23
model_name: yolo

hyperparameters:
  model_weights: yolo11n.pt
  epochs: 50
  imgsz: 640
  batch: 16
  device: "0"

coco_json: ../data/h23/instances_merged.json
split_json: ../data/h23/split.json
images_dir: ../data/h23/images
output_dir: ../outputs

wandb_project: freeman-bird-detection
visualize_split: test
```

**Key config fields:**
- `config_name` — used for naming output directories
- `model_name` — which adapter to use (`"yolo"`, `"rtdetr"`, or `"megadetector"`)
- `hyperparameters` — model-specific training parameters passed through to the adapter
- `wandb_project` — optional Weights & Biases project name for logging
- `visualize_split` — which split to visualize after prediction (`"train"`, `"val"`, or `"test"`)
- `resume_from` / `resume_experiment` — see [Resuming Training](#resuming-training) below

---

## Running Experiments

Each experiment produces a timestamped output directory under `outputs/<config_name>_<timestamp>/` containing:
- `config.json` — full experiment configuration
- `model.json` — paths to best and last checkpoint weights
- `metrics.json` — evaluation results
- `detections.json` — per-frame inference results
- `visualizations/` — annotated output videos
- `experiment.log` — full run log

### Full Pipeline

Run all stages (train → evaluate → predict → visualize) in sequence using `run_pipeline()`:

```python
from hlwdetector import ExperimentRunner

runner = ExperimentRunner("configs/yolo11_h23.yaml")
runner.run_pipeline()
```

### Running Stages Individually

Each stage can also be called separately. This is useful for re-running evaluation or visualization without retraining:

```python
from hlwdetector import ExperimentRunner

runner = ExperimentRunner("configs/yolo11_h23.yaml")
runner.train()
runner.evaluate()
runner.predict()
runner.visualize_predictions()
```

### Attaching to an Existing Experiment

To run evaluation, prediction, or visualization on a previously completed experiment, use `ExperimentRunner.from_experiment_dir()`. This attaches to the existing output directory — no new timestamped directory is created, and all outputs are written back into the original run.

```python
from hlwdetector import ExperimentRunner

runner = ExperimentRunner.from_experiment_dir("outputs/yolo11_h23_20260312_233336")
runner.evaluate()
runner.predict()
runner.visualize_predictions()
```

This requires that `config.json` and `model.json` are present in the experiment directory (i.e., training completed successfully).

### Resuming Training

To continue training from a prior checkpoint, set both `resume_from` and `resume_experiment` in the config. `resume_from` points to the model weights file; `resume_experiment` is the name of the original output directory. A new timestamped output directory is created for the resumed run.

```yaml
resume_experiment: yolo11_h23_20260402_004059
resume_from: ../outputs/yolo11_h23_20260402_004059/work/runs/yolo11_h23_train/weights/last.pt
```

Both fields must be set together or left unset.

### Running Multiple Experiments

To run several configs in sequence:

```python
from hlwdetector.runner import ExperimentRunner

for config in ["configs/yolo11_h23.yaml", "configs/yolo26_h23.yaml", "configs/rtdetr_h23.yaml"]:
    ExperimentRunner(config).run_pipeline()
```

---

## hlwdetector Package

### `config.py` — Experiment Configuration

`ExperimentConfig` is a dataclass that defines all parameters for an experiment. Configs are loaded from YAML files with all paths resolved relative to the YAML's location.

### `runner.py` — Experiment Runner

`ExperimentRunner` is the main entry point. It exposes:
- `run_pipeline()` — full train → evaluate → predict → visualize sequence
- `train()` — data preparation and model training
- `evaluate()` — evaluation on the configured split, writes `metrics.json`
- `predict()` — inference on the configured split, writes `detections.json`
- `visualize_predictions()` — generates annotated videos from predictions
- `ExperimentRunner.from_experiment_dir(path)` — attach to a completed experiment for post-hoc evaluation or visualization

### `adapters/` — Model Adapters

All models implement `BaseModelAdapter` from `adapters/base.py`:

```python
class BaseModelAdapter(ABC):
    def prepare_data(self) -> None: ...
    def train(self) -> TrainingResult: ...
    def evaluate(self) -> MetricsDict: ...
    def predict(self) -> DetectionResult: ...
```

Adapters are registered with `@register_adapter(name)` and resolved by `model_name` from the config.

**`yolo_adapter.py`** — `@register_adapter("yolo")`

Wraps Ultralytics YOLO models (YOLO11, YOLO26, etc.). `prepare_data()` converts COCO annotations to YOLO format and writes a `yolo.yaml` dataset config. `train()` runs Ultralytics training with hyperparameters from the config.

**`rtdetr_adapter.py`** — `@register_adapter("rtdetr")`

Wraps Ultralytics RT-DETR models. Same interface as the YOLO adapter.

**`megadetector_adapter.py`** — `@register_adapter("megadetector")`

Wraps MegaDetector V6 via the PytorchWildlife library. No fine-tuning is performed; the pretrained model runs inference directly. **Not fully implemented.**

### `dataset_manager.py` — Dataset Loading

`DatasetManager` loads the COCO JSON and split definition, producing per-split views of the data. Images are read from a single flat `images_dir`; split membership is determined at runtime by matching each frame's filename prefix against the video stems listed in `split.json`.

```python
from hlwdetector.dataset_manager import DatasetManager

dm = DatasetManager(config)
train_split = dm.get_split("train")
# train_split.images, train_split.annotations, train_split.image_paths
```

### `artifact_manager.py` — Output Artifacts

`ArtifactManager` manages all output paths and serialization for an experiment. Use `ArtifactManager.from_existing_dir(path)` (called automatically by `ExperimentRunner.from_experiment_dir`) to attach to a completed run without creating a new directory.

### `tracker.py` — Experiment Tracking

`ExperimentTracker` handles both local and Weights & Biases metric logging. W&B is optional and non-fatal if unavailable. Metrics are written to `metrics.json` on every `log()` call so that results are preserved if a job is preempted.

### `visualization/` — Output Videos

`VisualizationPipeline` generates annotated videos from the configured split, overlaying ground-truth boxes (green) and predicted boxes (blue). When a `frame_map.csv` is present, one video is produced per source video.

---

## Utilities

### `utilities/video_dataset_prep_tools.py`

- `extract_frames_from_dir(video_dir, output_dir)` — extracts all frames from `.mp4` files into a flat directory, writes `frame_map.csv`
- `stratified_video_split(coco_json, output_path, ...)` — creates a stratified train/val/test split by video prevalence quartile
- `extract_frames_by_split(video_dir, split_json, output_dir)` — extracts frames organized by split

### `utilities/annotation_converter.py`

`AnnotationConverter` converts between annotation formats (CVAT XML, COCO JSON, YOLO):

```python
from utilities.annotation_converter import AnnotationConverter

converter = AnnotationConverter()
converter.coco_to_yolo(
    coco_json="data/h23/instances_merged.json",
    output_dir="data/h23/yolo_labels",
    images_dir="data/h23/images"
)
```

---

## Adding a New Model

1. Create `hlwdetector/adapters/my_model_adapter.py`
2. Implement `BaseModelAdapter` and register it:
   ```python
   from hlwdetector.registry import register_adapter
   from hlwdetector.adapters.base import BaseModelAdapter

   @register_adapter("my_model")
   class MyModelAdapter(BaseModelAdapter):
       def prepare_data(self): ...
       def train(self): ...
       def evaluate(self): ...
       def predict(self): ...
   ```
3. Import the adapter in `hlwdetector/adapters/__init__.py` to trigger registration
4. Set `model_name: my_model` in a config YAML
