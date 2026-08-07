# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
Georgia Tech Mountain Bird Lab — automated bird detection in camera trap videos from the Freeman site. The repository is a full ML experimentation framework (`hlwdetector`) for training and evaluating multiple detection models (YOLO, RT-DETR, Swin Transformer) on camera trap footage.

> **Ignore the `archive/` directory.** It holds legacy code, models, and annotations that are no longer part of the framework (including the retired MegaDetector adapter). Do not read, edit, reference, or import from `archive/` when working on the active codebase.

## Repository Structure

```
hlwdetector/              # Core framework package
  config/
    experiment_config.py  # ExperimentConfig dataclass with YAML loading
    hpo_config.py         # HPOConfig / StudyArgs / OptimizeArgs for Optuna studies
  runner.py               # ExperimentRunner — main entry point
  hp_optimizer.py         # HPOptimizer — Optuna hyperparameter optimization
  registry.py             # Model adapter registry (@register_adapter)
  dataset_manager.py      # COCO dataset loading and split filtering
  artifact_manager.py     # Output paths and artifact serialization (experiments + HPO studies)
  tracker.py              # Experiment tracking (local + W&B)
  adapters/
    base.py               # BaseModelAdapter ABC
    yolo_adapter.py       # YOLO11/YOLO26 adapter
    rtdetr_adapter.py     # RT-DETR adapter
    swin_adapter.py       # Swin Transformer + Faster R-CNN adapter (timm + torchvision)
  visualization/
    pipeline.py           # VisualizationPipeline
    video_annotator.py    # Overlays GT + predictions and writes MP4

utilities/                # Data prep and annotation tools
  annotation_converter.py      # Multi-format converter: CVAT ↔ YOLO ↔ COCO
  video_dataset_prep_tools.py  # Frame extraction, stratified splitting
  visualization.py             # Additional visualization helpers

configs/                  # YAML configurations
  experiment/             # Single-experiment configs, one per model/dataset/run variant
                          # (e.g. <model>_<dataset>_<full|subset|resume>.yaml)
  hpo/                    # Optuna HPO study configs (e.g. <model>_hpo.yaml)

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
  hpo/
    <study>_<timestamp>/  # One directory per Optuna study
      trials.csv          # Per-trial hyperparameters, metric, state, duration
      study_summary.json  # Best trial number, config name, value, params
      optuna_journal.log  # Optuna journal storage (default backend)
      hpo.log             # Study-level log
                          # (each trial also writes its own <study>_trial_<n>_<timestamp>/ run dir)

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
ExperimentRunner("configs/experiment/yolo11_h23_full.yaml").run_pipeline()
```

### Run individual stages
```python
runner = ExperimentRunner("configs/experiment/yolo11_h23_full.yaml")
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

### Run a hyperparameter optimization study
```python
from hlwdetector.hp_optimizer import HPOptimizer

optimizer = HPOptimizer("configs/hpo/yolo26_hpo.yaml")
study = optimizer.run_study()   # returns the completed optuna.Study
```

### CLI
```bash
python -m hlwdetector.runner configs/experiment/yolo11_h23.yaml
```

## Config File Format

Relative path fields are resolved relative to the repository root (not the YAML file's location).

### Experiment config (`configs/experiment/`)

```yaml
config_name: yolo11_h23          # Unique identifier (used in output dir name)
model_name: yolo                  # Registered adapter name: yolo | rtdetr | swin
model_weights: yolo11n.pt         # Weights filename or path
hyperparameters:
  epochs: 50
  imgsz: 640
  batch: 32
  device: "0"
coco_json: data/h23/instances_merged.json
split_json: data/h23/split_h23.json
images_dir: data/h23/images
output_dir: outputs
wandb_project: freeman-bird-detection   # Optional
wandb_group: null                       # Optional (set automatically for HPO trials)
visualize_split: test                   # Optional (default: test)
visualization_fps: 29.0                 # Optional
# Resume training (both fields required together; their presence IS the resume flag):
# resume_experiment_name: yolo11_h23_20260416_083051   # dir NAME under <output_dir>/experiments/
# resume_weights: outputs/yolo11_h23_20260416_083051/work/runs/.../best.pt   # weights PATH
```

### HPO study config (`configs/hpo/`)

Consumed by `HPOptimizer` (via `HPOConfig`). Each trial is turned into an `ExperimentConfig` by merging `hyperparameters.static` with the sampled hyperparameters.

```yaml
model_name: yolo                  # Registered adapter name: yolo | rtdetr | swin
model_weights: yolo26n.pt         # Shared by every trial
metric: map50_95                  # MetricsDict field to optimize: precision | recall | f1 | map50 | map50_95
coco_json: data/h23/instances_subset.json
images_dir: data/h23/images
split_json: data/h23/split_h23_subset.json
output_dir: outputs
wandb_project: freeman-bird-detection   # Optional (all trials share a W&B group)
random_seed: 42
study_args:                       # kwargs for optuna.create_study
  study_name: yolo26_hpo
  direction: maximize             # maximize | minimize
  sampler: TPE                    # TPE | Random | Grid | CmaEs
  pruner: Hyperband               # Median | Hyperband | SuccessiveHalving | None
  storage: null                   # Optuna DB URL (e.g. sqlite:///hpo.db); null = journal file in study dir
optimize_args:                    # kwargs for study.optimize
  n_trials: 20
  timeout: null                   # Wall-clock limit in seconds (null = none)
hyperparameters:                  # Search space tiers: static | categorical | int | float
  static:                         # Fixed for every trial
    epochs: 3
    imgsz: 640
  categorical:                    # trial.suggest_categorical — value is the list of choices
    batch: [8, 16, 32]
    optimizer: [SGD, Adam, AdamW]
  int:                            # trial.suggest_int — [low, high, {**kwargs}]
  float:                          # trial.suggest_float — [low, high, {**kwargs}]
    lr0: [0.0001, 0.01, {log: True}]
```

Pruners read per-epoch metrics reported by the adapter's training callback; pruning is currently wired for the YOLO adapter.

## Architecture

### Adapter Pattern
Each model is wrapped in an adapter that implements `BaseModelAdapter`:
- `prepare_data(dataset_manager, config)` — convert dataset to model format
- `train(config)` — train or load pretrained weights
- `evaluate(config)` — evaluate on val split, return `MetricsDict`
- `predict(config)` — inference on test split, return `DetectionResult`

New adapters self-register via `@register_adapter("name")` and are imported through `adapters/__init__.py`.

### Swin Adapter Constraints
`swin_adapter.py` bridges timm and torchvision, which disagree on three conventions. Each is load-bearing — changing any one silently breaks training:

- **Resize/normalize belong to the model, not `prepare_data`.** `_COCODetectionDataset` returns native-resolution images (`ToTensor()` only) with boxes in native COCO pixel coordinates. Faster R-CNN's built-in `GeneralizedRCNNTransform` owns resize + normalization because it rescales the GT boxes in lockstep with the image, and un-scales predictions back to native coordinates during postprocessing. Resizing the image in `prepare_data` would leave boxes at the original scale — silently corrupt targets, plus `detections.json` in a different coordinate space than the GT that `VisualizationPipeline` draws.
- **Swin emits NHWC; FPN requires NCHW.** timm's `features_only` Swin returns channels-last tensors (e.g. `(B, 160, 160, 96)`). `_SwinBackboneWithFPN.forward` permutes per stage, gated on the known channel count so an NCHW backbone passes through unchanged.
- **`strict_img_size=False` is required.** `swin_*_224` asserts input matches the 224 baked into its name, but the detection transform emits variable, non-square batches (1280x720 at `imgsz: 640` → 640x1152 after `size_divisible=32` padding). The only surviving constraint is divisibility by the patch size (4), which that padding satisfies.

`imgsz` sets the **short side** of the network input with aspect ratio preserved (torchvision's `min_size` convention), not a square as in the YOLO/RT-DETR adapters — so `imgsz: 640` means 640x1152 here, not a 640x640 letterbox. Keep it at 640: h23 birds have a median √area of ~56px natively, which lands near the smallest anchor (32px) after this resize. Shrinking to 224 drops the median bird to ~13px, below every anchor, leaving the RPN with almost no positive targets.

### Key Classes
- **`ExperimentConfig`** — typed dataclass for all experiment config fields; validates prerequisites before run
- **`HPOConfig`** — typed dataclass for an Optuna study (search space + `study_args`/`optimize_args`); `validate()` checks adapter, paths, metric, direction, trial count, and search-space tiers
- **`ExperimentRunner`** — orchestrates a single experiment (train → evaluate → predict → visualize)
- **`HPOptimizer`** — drives an Optuna study, running one `ExperimentRunner` per trial and recording results; `run_study()` returns the `optuna.Study`
- **`DatasetManager`** — loads COCO JSON, filters images by video stem prefix against split.json lists
- **`ArtifactManager`** — creates timestamped output dirs for both experiments and HPO studies, serializes/deserializes all artifacts (including `trials.csv` and `study_summary.json`)
- **`ExperimentTracker`** — writes metrics.json after every log call; W&B failures are non-fatal; `wandb_group` groups a study's trials
- **`VisualizationPipeline`** — generates annotated MP4 with ground truth (green) and predictions (red) overlaid

### Data Flow
1. `DatasetManager` loads COCO JSON and partitions images into train/val/test `SplitView` objects
2. Adapter's `prepare_data()` converts to model-specific format (e.g., YOLO .txt labels + dataset YAML)
3. Training writes checkpoints to `work/runs/`; best.pt is retained for evaluation
4. Predictions stored as `Dict[frame_stem, sv.Detections]`, serialized to `detections.json`
5. `VisualizationPipeline` reads detections and GT to produce annotated video

### Resume Flow
Setting `resume_experiment_name` + `resume_weights` is itself the signal to resume — there is no separate
opt-in flag, and the two fields hold different kinds of value:

- **`resume_experiment_name`** — the original run's directory *name*. `ExperimentConfig.resume_experiment_dir`
  is the single place that expands it to `<output_dir>/experiments/<name>`; never join that path by hand.
  It locates the two things the checkpoint can't supply: the prepared data yaml in `work/`
  (`_discover_data_yaml`) and the `config.json` that gets a `resumed_in` breadcrumb plus the
  `wandb_run_id` to rejoin.
- **`resume_weights`** — the *path* to the weights the adapter loads. It is in `PATH_FIELDS`; `resume_experiment_name`
  deliberately is not.

What a resume reuses vs. creates:
- **Output dir — new.** `ArtifactManager` always mints `<config_name>_<timestamp>`. The original run is
  untouched apart from `resumed_in`, which is the only link between the two dirs.
- **W&B run — the original.** `ExperimentTracker` reads `wandb_run_id` from the *original* dir's
  `config.json` and inits with `resume="must"`, so metrics continue one history. The run keeps its
  original name even though artifacts land in the new dir. The id is copied into the new dir's
  `config.json`, so resume-of-a-resume chains. Missing id → warning, new run.
- **Optimizer/scheduler/epoch — restored** from the checkpoint (swin adapter).

`hyperparameters.epochs` is an **absolute** target, not a count of additional epochs: swin checkpoints
store epochs *completed* and the loop runs `range(start_epoch, epochs)`. Resuming a 3-epoch checkpoint
with `epochs: 3` trains nothing, so the adapter raises rather than silently no-op.

### HPO Flow
1. `HPOptimizer` loads an `HPOConfig` and creates an `ArtifactManager` for the study dir (`outputs/hpo/<study>_<timestamp>/`)
2. `run_study()` builds the Optuna sampler/pruner/storage from the config's name strings and calls `study.optimize()`
3. Each trial samples the search space, merges it with `hyperparameters.static` into an `ExperimentConfig`, and runs a fresh `ExperimentRunner.train()` + `evaluate()`
4. The adapter's per-epoch callback reports metrics to Optuna so the pruner can stop losing trials early (YOLO adapter)
5. Trial results are appended to `trials.csv`; the best trial is written to `study_summary.json`

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
- **optuna** — hyperparameter optimization (samplers, pruners, journal/DB storage)
- **torch/torchvision** — deep learning framework; also provides Faster R-CNN head for Swin adapter
- **opencv-python** — video and frame I/O

## Data Sources
- Raw videos: Freeman Bird Lab Cameratrap Videos downloaded from various sources
- Annotations: COCO JSON downloaded with corresponding videos