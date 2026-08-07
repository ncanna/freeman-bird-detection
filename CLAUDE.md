# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
Georgia Tech Mountain Bird Lab — automated bird detection in camera trap videos from the Freeman site. The repository is a full ML experimentation framework (`hlwdetector`) for training and evaluating multiple detection models (YOLO, RT-DETR, Swin Transformer, DETR) on camera trap footage.

> **Ignore the `archive/` directory.** It holds legacy code, models, and annotations that are no longer part of the framework (including the retired MegaDetector adapter). Do not read, edit, reference, or import from `archive/` when working on the active codebase.
>
> **Ignore the `detr_detector/` directory too.** It is a vendored copy of the original Facebook DETR research repo, kept for reference only. It is standalone and not wired into `hlwdetector` — the active DETR support is `hlwdetector/adapters/detr_adapter.py`, built on HuggingFace Transformers.

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
  results.py              # MetricsDict / TrainingResult / DetectionResult dataclasses
  metrics.py              # compute_coco_metrics — shared pycocotools COCOeval scoring
  paths.py                # Repo-root anchoring for portable artifact paths
  adapters/
    base.py               # BaseModelAdapter ABC + resolve_device (re-exports result types)
    yolo_adapter.py       # YOLO11/YOLO26 adapter
    rtdetr_adapter.py     # RT-DETR adapter
    swin_adapter.py       # Swin Transformer + Faster R-CNN adapter (timm + torchvision)
    detr_adapter.py       # DETR-family adapter (HuggingFace Transformers)
  visualization/
    pipeline.py           # VisualizationPipeline
    video_annotator.py    # Overlays GT + predictions and writes MP4
    confusion_matrix.py   # ConfusionMatrixVisualizer — frame-level TP/FP/TN/FN
    metrics_comparator.py # MetricsComparator — compare metrics across runs

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
    split.json                  # Train/val/test video stem lists (full)
    split_h23_subset.json       # Train/val/test video stem lists (subset)
  h03/                    # H03 camera trap dataset
  african-wildlife/       # Reference dataset

outputs/
  experiments/            # Experiment results (one directory per run)
    <config>_<timestamp>/
      config.json         # Saved ExperimentConfig
      model.json          # Checkpoint paths (stored repo-relative)
      metrics.json        # Evaluation metrics (updated progressively)
      detections.json     # Per-frame predictions
      experiment.log      # Full logging output
      work/               # Training artifacts (YOLO: train.txt/yolo.yaml; torch: runs/train/weights/)
      visualizations/     # Annotated MP4 video outputs
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
detr_detector/            # Vendored Facebook DETR research repo — IGNORE (not wired in)
run_experiments.py        # Example runner script
run_hpo.py                # Example HPO study script
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
runner = ExperimentRunner.from_experiment_dir("outputs/experiments/yolo11_h23_20260416_083051")
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
model_name: yolo                  # Registered adapter name: yolo | rtdetr | swin | detr
model_weights: yolo11n.pt         # Weights filename, path, or HF hub id (detr)
hyperparameters:
  epochs: 50
  imgsz: 640
  batch: 32
  device: "0"
coco_json: data/h23/instances_merged.json
split_json: data/h23/split.json
images_dir: data/h23/images
output_dir: outputs
wandb_project: freeman-bird-detection   # Optional
wandb_group: null                       # Optional (set automatically for HPO trials)
visualize_split: test                   # Optional (default: test)
visualization_fps: 29.0                 # Optional
# Resume training (both fields required together; their presence IS the resume flag):
# resume_experiment_name: yolo11_h23_20260416_083051   # dir NAME under <output_dir>/experiments/
# resume_weights: outputs/experiments/yolo11_h23_20260416_083051/work/runs/.../best.pt   # weights PATH
```

`model_weights` is **not** a path field, so it is never resolved or checked against the
filesystem. That is what lets the DETR adapter pass a HuggingFace hub id
(`facebook/detr-resnet-50`) through the same field the YOLO adapters use for a `.pt` file.

### HPO study config (`configs/hpo/`)

Consumed by `HPOptimizer` (via `HPOConfig`). Each trial is turned into an `ExperimentConfig` by merging `hyperparameters.static` with the sampled hyperparameters.

```yaml
model_name: yolo                  # Registered adapter name: yolo | rtdetr | swin | detr
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

### DETR Adapter Constraints
`detr_adapter.py` is **checkpoint-driven**: `config.model_weights` names any HuggingFace object-detection
checkpoint (`facebook/detr-resnet-50`, `SenseTime/deformable-detr`,
`microsoft/conditional-detr-resnet-50`, …) and `AutoModelForObjectDetection` / `AutoImageProcessor` pick
the right classes. Switching variants is a YAML edit, not a new adapter.

Three conventions are load-bearing. Changing any one silently breaks training or misplaces boxes:

- **Resize/normalize belong to the image processor, not `prepare_data`.** Same reasoning as swin: the
  dataset yields native-resolution PIL images with boxes in native COCO pixel coordinates, and
  `DetrImageProcessor` rescales boxes in lockstep with the image.
  `post_process_object_detection(target_sizes=...)` maps predictions back to native coordinates, which
  is the space `VisualizationPipeline` draws in. Resizing in `prepare_data` would leave boxes at the
  original scale.
- **`imgsz` is the short side, not a square** — the torchvision/DETR convention, matching the swin
  adapter and unlike YOLO/RT-DETR. `imgsz: 640` on 1280x720 footage means a 640x1138 network input.
  The long side defaults to `imgsz * 16/9` and is overridable via `hyperparameters.max_size`.
- **COCO `category_id` must be remapped to a contiguous 0-based class index.** HF DETR reserves the
  *last* logit for "no object", so with `num_labels=1` the only valid class label is `0`. Feeding h23's
  raw `category_id: 1` trains every box as background **with no error raised**. Never add a background
  class manually — unlike torchvision's Faster R-CNN, which does reserve index 0 for background.

Two further notes specific to this adapter:

- **`ignore_mismatched_sizes=True` is required** when loading a pretrained checkpoint: the hub weights
  ship a 91-class COCO head that must be swapped for the dataset's class count.
- **`epochs` needs to be large.** DETR converges slowly — the original paper trains 300–500 epochs. At
  2 epochs on h23 the peak prediction confidence is ~0.0025, so the default `score_threshold: 0.5`
  yields an empty `detections.json` and a video with no prediction boxes. That is under-training, not
  a bug. `SenseTime/deformable-detr` converges substantially faster if that matters more than
  fidelity to vanilla DETR.

### Metrics
`hlwdetector/metrics.py` holds `compute_coco_metrics(split_view, detections, max_dets=100)`, the single
scoring path for the adapters that do not get metrics from their training framework (**swin** and
**detr**; YOLO and RT-DETR still read Ultralytics' own numbers off `results.box`).

- Ground truth is built **in memory from the `SplitView`**, not from `coco_json` on disk — a split is a
  subset of that file, so evaluating against the whole file would count every other split's images as
  missed detections.
- `map50_95` is `stats[0]` (AP@[.50:.95]) and `map50` is `stats[1]` (AP@0.50). Scalar
  `precision`/`recall`/`f1` are the **best-F1 point on the IoU=0.50 PR curve**, which matches the
  semantics of Ultralytics' `box.mp`/`box.mr` and keeps the numbers comparable across adapters.
- `raw` carries all 12 COCO summary statistics. Note that h23 birds (median √area ~56px → area ~3.1k)
  land in COCO's **medium** bucket (32²–96²), so `APm` is the size band to watch; `APs` reports `-1`
  because the dataset contains no small objects by COCO's definition.
- **Callers must pass detections at a low threshold.** Adapters read `hyperparameters.eval_score_threshold`
  (default `0.001`) for metrics, separate from `score_threshold` (default `0.5`) used for `predict()`
  and visualization. AP is the area under the full PR curve; pre-filtering at 0.5 truncates it and
  systematically understates AP.
- `coco_gt.loadRes([])` raises `IndexError`, so the empty-detection case is short-circuited to zeros —
  an untrained model at epoch 1 legitimately produces nothing.

Result dataclasses live in `hlwdetector/results.py`, one layer **below** the adapters, so `metrics.py`
can build a `MetricsDict` without importing the adapters package back (importing `adapters.base` runs
`adapters/__init__.py`, which imports every adapter, which imports `metrics` — a cycle).
`adapters/base.py` re-exports all three names, so `from hlwdetector.adapters.base import MetricsDict`
keeps working and returns the same object.

### Key Classes
- **`ExperimentConfig`** — typed dataclass for all experiment config fields; validates prerequisites before run
- **`HPOConfig`** — typed dataclass for an Optuna study (search space + `study_args`/`optimize_args`); `validate()` checks adapter, paths, metric, direction, trial count, and search-space tiers
- **`ExperimentRunner`** — orchestrates a single experiment (train → evaluate → predict → visualize)
- **`HPOptimizer`** — drives an Optuna study, running one `ExperimentRunner` per trial and recording results; `run_study()` returns the `optuna.Study`
- **`DatasetManager`** — loads COCO JSON, filters images by video stem prefix against split.json lists
- **`ArtifactManager`** — creates timestamped output dirs for both experiments and HPO studies, serializes/deserializes all artifacts (including `trials.csv` and `study_summary.json`)
- **`ExperimentTracker`** — writes metrics.json after every log call; W&B failures are non-fatal; `wandb_group` groups a study's trials
- **`VisualizationPipeline`** — generates annotated MP4 with ground truth (green) and predictions (blue) overlaid
- **`compute_coco_metrics`** (`metrics.py`) — shared pycocotools COCOeval scoring for the swin and detr adapters

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
- **Optimizer/scheduler/epoch — restored** from the checkpoint (swin and detr adapters).

`hyperparameters.epochs` is an **absolute** target, not a count of additional epochs: swin and detr
checkpoints store epochs *completed* and the loop runs `range(start_epoch, epochs)`. Resuming a 3-epoch
checkpoint with `epochs: 3` trains nothing, so the adapter raises rather than silently no-op.

**`ExperimentRunner.train()` skips `prepare_data` entirely when `resume_weights` is set**, so an adapter
that needs its datasets during training (e.g. detr's per-epoch validation pass) must rebuild them
itself. `DETRAdapter._ensure_datasets` does this by constructing its own `DatasetManager(config)` and
calling `prepare_data`. **The swin adapter does not**, so its `_val_dataset`/`_val_split` stay `None` on
a resumed run and a subsequent `evaluate()`/`predict()` in the same process will fail — a known gap.

The detr adapter also carries `best_metric` **across** the resume boundary (a resumed run is a
continuation, so an epoch counts as "best" only if it beats the whole history) and seeds `best.pt` from
the resumed checkpoint so `best_weights_path` is always populated. `training_metrics.best_epoch == -1`
means no epoch in the new run beat the checkpoint it resumed from.

### HPO Flow
1. `HPOptimizer` loads an `HPOConfig` and creates an `ArtifactManager` for the study dir (`outputs/hpo/<study>_<timestamp>/`)
2. `run_study()` builds the Optuna sampler/pruner/storage from the config's name strings and calls `study.optimize()`
3. Each trial samples the search space, merges it with `hyperparameters.static` into an `ExperimentConfig`, and runs a fresh `ExperimentRunner.train()` + `evaluate()`
4. The adapter's per-epoch callback reports metrics to Optuna so the pruner can stop losing trials early (YOLO adapter)
5. Trial results are appended to `trials.csv`; the best trial is written to `study_summary.json`

**Pruning only works for the YOLO adapter.** `hp_optimizer._METRIC_KEY_MAP` hardcodes Ultralytics key
strings (`metrics/mAP50-95(B)`, …), and `hp_optimizer` assigns `adapter._hpo_pruning_callback`
unconditionally, so an rtdetr/swin/detr study configured with a pruner sets an attribute that either
nobody reads or that reports keys Optuna cannot match — no error, no pruning. The detr adapter defines
and invokes the hook so the wiring exists, but it emits `val/mAP50_95`-style keys, not Ultralytics ones.
See `docs/plans/hpo-pruning-rtdetr-swin.md`.

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
- **transformers (>=5.0)** — DETR-family models and image processors. The v5 vision API differs from
  the widely-circulated v4 tutorials (`AutoImageProcessor` now returns the `*Fast` variant by default,
  and processor signatures moved), so verify against the installed package rather than writing from
  memory of v4.
- **timm** — Swin Transformer backbone with multi-scale feature extraction
- **supervision** — `sv.Detections` used as the standard detection container throughout
- **pycocotools** — COCO evaluation (`COCOeval`), via `hlwdetector/metrics.py`
- **wandb** — experiment tracking (optional)
- **optuna** — hyperparameter optimization (samplers, pruners, journal/DB storage)
- **torch/torchvision** — deep learning framework; also provides Faster R-CNN head for Swin adapter
- **opencv-python** — video and frame I/O

## Data Sources
- Raw videos: Freeman Bird Lab Cameratrap Videos downloaded from various sources
- Annotations: COCO JSON downloaded with corresponding videos