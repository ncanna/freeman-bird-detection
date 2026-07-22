# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

<<<<<<< Updated upstream
Georgia Tech Mountain Bird Lab — automated bird detection in camera trap videos from the Freeman site. The repository is a full ML experimentation framework (`hlwdetector`) for training and evaluating detection models (YOLO, RT-DETR) on camera trap footage.

> **Ignore the `archive/` directory.** It holds legacy code, models, and annotations that are no longer part of the framework (including the retired MegaDetector adapter). Do not read, edit, reference, or import from `archive/` when working on the active codebase.
=======
Georgia Tech Mountain Bird Lab — automated bird detection in camera trap videos from the Freeman site. The repository is a full ML experimentation framework (`hlwdetector`) for training and evaluating multiple detection models (YOLO, RT-DETR, Swin Transformer) on camera trap footage.
>>>>>>> Stashed changes

## Repository Structure

```
hlwdetector/              # Core framework package
  config.py               # ExperimentConfig dataclass with YAML loading
  runner.py               # ExperimentRunner — main entry point
  registry.py             # Model adapter registry (@register_adapter)
  dataset_manager.py      # COCO dataset loading and split filtering
  artifact_manager.py     # Output paths and artifact serialization
  tracker.py              # Experiment tracking (local + W&B)
  adapters/
    base.py               # BaseModelAdapter ABC
    yolo_adapter.py       # YOLO11/YOLO26 adapter
    rtdetr_adapter.py     # RT-DETR adapter
<<<<<<< Updated upstream
=======
    swin_adapter.py       # Swin Transformer + Faster R-CNN adapter (timm + torchvision)
    megadetector_adapter.py  # MegaDetector V6 (pretrained, no fine-tuning)
>>>>>>> Stashed changes
  visualization/
    pipeline.py           # VisualizationPipeline
    video_annotator.py    # Overlays GT + predictions and writes MP4

utilities/                # Data prep and annotation tools
  annotation_converter.py      # Multi-format converter: CVAT ↔ YOLO ↔ COCO
  video_dataset_prep_tools.py  # Frame extraction, stratified splitting
  visualization.py             # Additional visualization helpers

configs/                  # YAML experiment configurations, one per model/dataset/run variant
                          # (e.g. <model>_<dataset>_<full|subset|resume>.yaml)

data/
  h23/                    # Main dataset (extracted frames + COCO annotations)
    images/               # Flat directory of PNG frames
    labels/               # YOLO-format label .txt files
    instances_merged.json       # Full COCO annotations
    instances_subset.json       # Subset COCO annotations
    split_h23.json              # Train/val/test video stem lists (full)
    split_h23_subset.json       # Train/val/test video stem lists (subset)
  h03/                    # H03 camera trap dataset
  african-wildlife/       # Reference dataset

outputs/                  # Experiment results (one directory per run)
  <config>_<timestamp>/
    config.json           # Saved ExperimentConfig
    model.json            # Checkpoint paths
    metrics.json          # Evaluation metrics (updated progressively)
    detections.json       # Per-frame predictions
    experiment.log        # Full logging output
    work/                 # YOLO training artifacts (train.txt, yolo.yaml, runs/)
    visualizations/       # Annotated MP4 video outputs

docs/
  diagrams/               # PlantUML architecture diagrams (context, component, class)
models/                   # Pre-trained model weights
notebooks/                # Jupyter notebooks
archive/                  # Legacy code — IGNORE (not part of the active framework)
run_experiments.py        # Example runner script
run_experiments.ipynb     # Notebook runner
```

## Development Commands

### Setup
```bash
pip install -r requirements.txt
```

### Run full pipeline (train → eval → predict → visualize)
```python
from hlwdetector.runner import ExperimentRunner
ExperimentRunner("configs/yolo11_h23.yaml").run_pipeline()
```

### Run individual stages
```python
runner = ExperimentRunner("configs/yolo11_h23.yaml")
runner.train()
runner.evaluate()
runner.predict()
runner.visualize_predictions()
```

### Attach to an existing experiment (post-hoc eval/viz)
```python
runner = ExperimentRunner.from_experiment_dir("outputs/yolo11_h23_20260416_083051")
runner.evaluate()
runner.visualize_predictions()
```

### CLI
```bash
python -m hlwdetector.runner configs/yolo11_h23.yaml
```

## Config File Format

All fields with relative paths are resolved relative to the YAML file location.

```yaml
config_name: yolo11_h23          # Unique identifier (used in output dir name)
<<<<<<< Updated upstream
model_name: yolo                  # Registered adapter name: yolo | rtdetr
=======
model_name: yolo                  # Registered adapter name: yolo | rtdetr | swin
>>>>>>> Stashed changes
hyperparameters:
  model_weights: yolo11n.pt       # Weights filename or path
  epochs: 50
  imgsz: 640
  batch: 32
  device: cuda
coco_json: ../data/h23/instances_merged.json
split_json: ../data/h23/split_h23.json
images_dir: ../data/h23/images
output_dir: ../outputs
wandb_project: freeman-bird-detection   # Optional
visualize_split: test                   # Optional (default: test)
visualization_fps: 29.0                 # Optional
# Resume training (both fields required together):
# resume_experiment: yolo11_h23_20260416_083051
# resume_from: outputs/yolo11_h23_20260416_083051/work/runs/.../best.pt
```

## Architecture

### Adapter Pattern
Each model is wrapped in an adapter that implements `BaseModelAdapter`:
- `prepare_data(dataset_manager, config)` — convert dataset to model format
- `train(config)` — train or load pretrained weights
- `evaluate(config)` — evaluate on val split, return `MetricsDict`
- `predict(config)` — inference on test split, return `DetectionResult`

New adapters self-register via `@register_adapter("name")` and are imported through `adapters/__init__.py`.

### Key Classes
- **`ExperimentConfig`** — typed dataclass for all config fields; validates prerequisites before run
- **`DatasetManager`** — loads COCO JSON, filters images by video stem prefix against split.json lists
- **`ArtifactManager`** — creates timestamped output dirs, serializes/deserializes all artifacts
- **`ExperimentTracker`** — writes metrics.json after every log call; W&B failures are non-fatal
- **`VisualizationPipeline`** — generates annotated MP4 with ground truth (green) and predictions (red) overlaid

### Data Flow
1. `DatasetManager` loads COCO JSON and partitions images into train/val/test `SplitView` objects
2. Adapter's `prepare_data()` converts to model-specific format (e.g., YOLO .txt labels + dataset YAML)
3. Training writes checkpoints to `work/runs/`; best.pt is retained for evaluation
4. Predictions stored as `Dict[frame_stem, sv.Detections]`, serialized to `detections.json`
5. `VisualizationPipeline` reads detections and GT to produce annotated video

### Split Format (split.json)
```json
{
  "train": ["IMG_0050", "IMG_0065"],
  "val":   ["IMG_0019", "IMG_0032"],
  "test":  ["IMG_0074", "IMG_0077"]
}
```
Lists video stems; frames are matched at runtime by filename prefix.

## Key Dependencies
- **ultralytics** — YOLO and RT-DETR model training and inference
- **timm** — Swin Transformer backbone with multi-scale feature extraction
- **supervision** — `sv.Detections` used as the standard detection container throughout
- **pycocotools** — COCO evaluation (COCOeval)
- **wandb** — experiment tracking (optional)
- **torch/torchvision** — deep learning framework; also provides Faster R-CNN head for Swin adapter
- **opencv-python** — video and frame I/O

## Data Sources
- Raw videos: Freeman Bird Lab Cameratrap Videos downloaded from various sources
- Annotations: COCO JSON downloaded with corresponding videos