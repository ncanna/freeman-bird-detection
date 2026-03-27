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

## hlwdetector Package

The `hlwdetector` package provides the framework for defining, running, and tracking detection experiments.

### `config.py` — Experiment Configuration

`ExperimentConfig` is a dataclass that defines all parameters for an experiment. Configs are loaded from YAML files with all paths resolved relative to the YAML's location.

Key fields:
- `model_name` — which adapter to use (`"yolo11"` or `"megadetector"`)
- `experiment_name` — used for naming output directories
- `coco_json` — path to the COCO-format annotation file
- `images_dir` — root directory containing extracted frame images
- `split_json` — path to the train/val/test split definition
- `hyperparameters` — model-specific training parameters (passed through to the adapter)
- `output_dir` — root directory for all experiment outputs (default: `outputs/`)
- `wandb_project` — optional Weights & Biases project name
- `resume_from` — path to a prior experiment directory to resume from
- `visualize` / `visualize_split` — whether to produce annotated output videos

### `runner.py` — Experiment Runner

`ExperimentRunner` is the main entry point. It orchestrates the full pipeline: data preparation, training, evaluation, prediction, and visualization.

```python
from hlwdetector import ExperimentRunner

runner = ExperimentRunner("configs/yolo11_h23.yaml")
runner.run()
```

`runner.run()` executes:
1. `adapter.prepare_data()` — converts the dataset to the model's native format
2. `adapter.train()` — trains (or loads) the model
3. `adapter.evaluate()` — evaluates on the test split
4. `adapter.predict()` — runs inference on the test split
5. Visualization pipeline (if enabled)
6. Saves all artifacts and logs metrics

When `resume_from` is set in the config, steps 1 and 2 are skipped and weights are loaded from the prior run's artifacts.

### `adapters/` — Model Adapters

All models implement `BaseModelAdapter` from `adapters/base.py`:

```python
class BaseModelAdapter(ABC):
    def prepare_data(self) -> None: ...
    def train(self) -> TrainingResult: ...
    def evaluate(self) -> MetricsDict: ...
    def predict(self) -> DetectionResult: ...
```

Adapters are registered with `@register_adapter(name)` and discovered by name from the config.

**`yolo_adapter.py`** — `@register_adapter("yolo11")`

Wraps the Ultralytics YOLO11 model. `prepare_data()` converts COCO annotations to YOLO format and writes a `yolo.yaml` dataset config. `train()` runs Ultralytics training with hyperparameters from the config. Evaluation and prediction use `model.val()` and `model.predict()` respectively.

Supported hyperparameters:
| Parameter | Description | Default |
|-----------|-------------|---------|
| `model_weights` | Starting weights (e.g. `yolo11n.pt`) | required |
| `epochs` | Number of training epochs | required |
| `imgsz` | Input image size | required |
| `batch` | Batch size | required |
| `device` | CUDA device index or `"cpu"` | required |

**`megadetector_adapter.py`** — `@register_adapter("megadetector")`

Wraps MegaDetector V6 via the PytorchWildlife library. No fine-tuning is performed; the pretrained model runs inference directly. All detections are treated as birds (a species classifier is a planned future addition). Evaluation uses `pycocotools.COCOeval`.

Supported hyperparameters:
| Parameter | Description | Default |
|-----------|-------------|---------|
| `version` | MegaDetector version string | `"MDV6b-yolov9c"` |
| `confidence_threshold` | Detection confidence cutoff | `0.1` |

### `dataset_manager.py` — Dataset Loading

`DatasetManager` loads the COCO JSON and split definition, providing per-split views of the dataset.

```python
from hlwdetector.dataset_manager import DatasetManager

dm = DatasetManager(config)
train_split = dm.get_split("train")
# train_split.images, train_split.annotations, train_split.images_split_dir
```

The `split.json` file maps split names to lists of video stems:
```json
{
  "train": ["video_001", "video_002"],
  "val":   ["video_003"],
  "test":  ["video_004"]
}
```

### `artifact_manager.py` — Output Artifacts

`ArtifactManager` manages all output paths and serialization for an experiment run. It creates a timestamped directory under `output_dir/runs/<experiment_name>_<timestamp>/` and writes:
- `config.json` — full experiment configuration
- `model.json` — paths to best and last checkpoint weights
- `metrics.json` — evaluation results (updated on every log for preemption resilience)
- `detections.json` — per-frame inference results

### `tracker.py` — Experiment Tracking

`ExperimentTracker` handles both local and Weights & Biases metric logging. W&B is optional and non-fatal if unavailable. Metrics are written to `metrics.json` on every `log()` call so that results are preserved if a job is preempted on PACE ICE.

### `visualization/` — Output Videos

`VisualizationPipeline` (in `visualization/pipeline.py`) generates annotated videos from the test split, overlaying ground-truth boxes (green) and predicted boxes (blue). When a `frame_map.csv` is present, one video is produced per source video.

`VideoAnnotator` (in `visualization/annotator.py`) handles the frame-level rendering using the Roboflow Supervision library.

---

## Running Experiments

### Prerequisites

1. Extract video frames:
```python
from utilities.video_dataset_prep_tools import extract_frames_by_split

extract_frames_by_split(
    video_dir="/path/to/videos",
    split_json="data/h23/split.json",
    output_dir="data/h23/images"
)
```

2. Ensure you have:
   - A COCO-format annotation JSON (`instances_merged.json`)
   - A split JSON (`split.json`) mapping split names to video stems
   - Extracted frame images organized by split under `images_dir`

### Defining a Config

Create a YAML file in `configs/`. All paths are resolved relative to the YAML file's location:

```yaml
model_name: yolo11
experiment_name: yolo11_h23

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
visualize: true
```

### Running an Experiment

**From a script:**
```python
from hlwdetector import ExperimentRunner

runner = ExperimentRunner("configs/yolo11_h23.yaml")
runner.run()
```

**From the notebook:**

Open `run_experiments.ipynb` and run the cells. The notebook documents required paths and prints split statistics before launching the experiment.

### Resuming a Run

To resume a previously interrupted experiment, set `resume_from` in the config to the prior run's output directory. The runner will skip data preparation and training, loading weights from `model.json` in that directory.

```yaml
resume_from: ../outputs/runs/yolo11_h23_20260312_233336
```

---

## Utilities

### `utilities/video_dataset_prep_tools.py`

Tools for preparing video datasets:

- `extract_frames_from_dir(video_dir, output_dir)` — extracts all frames from `.mp4` files, writes `frame_map.csv`
- `stratified_video_split(coco_json, output_path, ...)` — creates a stratified train/val/test split by video prevalence quartile
- `extract_frames_by_split(video_dir, split_json, output_dir)` — extracts frames organized by split

### `utilities/annotation_converter.py`

`AnnotationConverter` converts between annotation formats:
- CVAT XML (track-based)
- COCO JSON
- YOLO (normalized center coordinates)

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
