# RT-DETR & YOLO Training Settings — h23 "default" runs

The detailed settings below are a **historical reference** taken from each run's
Ultralytics `work/runs/train/args.yaml`:

- **RT-DETR**: `outputs/rtdetr_h23_20260707_072734` (amp-on, 100 epochs, H200)
- **YOLO11**: `outputs/yolo11_h23_20260609_160816`

**Key point:** these runs use **Ultralytics defaults for essentially everything**. The
project YAML configs only set `model_weights`, `epochs`, `imgsz`, `batch`, `device`,
and `amp` — so RT-DETR and YOLO share **identical** settings. The only real differences
are the model weights and the dataloader `workers` (4 vs 8). Nothing in the augmentation
or optimizer space was actually tuned. The current full H23 configs now make the
study-critical defaults explicit and use 50 epochs: YOLO11/YOLO26 pin the current
effective `MuSGD`/`lr0=0.01` behavior, while RT-DETR uses `AdamW`/`lr0=0.0001`.

---

## 1. Historical core hyperparameters

| Parameter | RT-DETR | YOLO11 | Meaning |
|---|---|---|---|
| `model` | rtdetr-l.pt | yolo11m.pt | pretrained weights (COCO) |
| `epochs` | 100 | 100 | training epochs |
| `batch` | 32 | 32 | batch size |
| `imgsz` | 640 | 640 | input resolution |
| `amp` | true | true | mixed precision |
| `patience` | 100 | 100 | early-stop patience (= effectively no early stop) |
| `freeze` | None | None | no frozen layers |
| `pretrained` | true | true | start from pretrained weights |
| `rect` | False | False | rectangular training off |
| `multi_scale` | 0.0 | 0.0 | multi-scale training off |
| `dropout` | 0.0 | 0.0 | — |
| `deterministic` | True | True | deterministic ops |
| `seed` | 0 | 0 | RNG seed |
| `workers` | **4** | **8** | dataloader workers (only true non-model diff) |
| `close_mosaic` | 10 | 10 | disable mosaic for last N epochs |

---

## 2. Optimizer & LR schedule — likely the RT-DETR problem

| Parameter | Historical runs | Current 50-epoch study | Notes |
|---|---|---|---|
| `optimizer` | `auto` (SGD-family) | YOLO: `MuSGD`; RT-DETR: `AdamW` | Current Ultralytics `auto` ignores requested `lr0` and selects `MuSGD` for this study's iteration count, so the study pins optimizers explicitly |
| `lr0` | 0.01 effective | YOLO: 0.01; RT-DETR: 0.0001 | initial learning rate |
| `lrf` | 0.01 | 0.01 | final LR = `lr0 * lrf` |
| `cos_lr` | False | False | linear decay (not cosine) |
| `momentum` | 0.937 stored; 0.9 effective under `auto` | 0.9 | SGD momentum or AdamW beta1 |
| `weight_decay` | 0.0005 | 0.0005 | retained as the Ultralytics default and tested separately |
| `warmup_epochs` | 3.0 | 3.0 | |
| `warmup_momentum` | 0.8 | 0.8 | |
| `warmup_bias_lr` | 0.1 stored; 0.0 effective under `auto` | 0.0 | |
| `nbs` | 64 | 64 | nominal batch size for LR scaling |

**Why RT-DETR changed:** its prior SGD-family optimization at `lr0=0.01` is a poor
fit for a DETR/transformer model. The official RT-DETR recipe uses **AdamW at 1e-4**
(and a lower backbone LR that the simple Ultralytics argument surface cannot express).
The project therefore uses explicit `AdamW`/`lr0=0.0001` for fresh RT-DETR runs. This
also ensures `lr0` is honored instead of silently replaced by `optimizer=auto`.

---

## 3. Loss weights (identical, all defaults)

| `box` | `cls` | `dfl` | `pose` | `kobj` |
|---|---|---|---|---|
| 7.5 | 0.5 | 1.5 | 12.0 | 1.0 |

(`pose`/`kobj` are irrelevant for plain detection.)

---

## 4. Data augmentation (main focus) — identical for RT-DETR & YOLO

Every parameter from the [Ultralytics augmentation settings](https://docs.ultralytics.com/usage/cfg#augmentation-settings),
with the value used in these runs:

| Parameter | Value | On? | What it does |
|---|---|---|---|
| `hsv_h` | 0.015 | yes | hue jitter (fraction of the color wheel) |
| `hsv_s` | 0.7 | yes | saturation jitter (+/-70%) |
| `hsv_v` | 0.4 | yes | brightness/value jitter (+/-40%) |
| `degrees` | 0.0 | no | random rotation (degrees) |
| `translate` | 0.1 | yes | random shift (+/-10% of image size) |
| `scale` | 0.5 | yes | random zoom (+/-50%) |
| `shear` | 0.0 | no | shear (degrees) |
| `perspective` | 0.0 | no | perspective warp |
| `flipud` | 0.0 | no | vertical flip probability |
| `fliplr` | 0.5 | yes | horizontal flip probability |
| `bgr` | 0.0 | no | RGB->BGR channel swap probability |
| `mosaic` | 1.0 | yes | 4-image mosaic (always on) |
| `mixup` | 0.0 | no | blend two images together |
| `cutmix` | 0.0 | no | paste a rectangular patch from another image |
| `copy_paste` | 0.0 | no | copy objects between images (needs segmentation masks) |
| `copy_paste_mode` | flip | n/a | only used when `copy_paste > 0` |
| `auto_augment` | randaugment | classification-only | RandAugment policy; NOT applied to detection training |
| `erasing` | 0.4 | classification-only | random erasing; NOT applied to detection training |
| `crop_fraction` | (absent) | n/a | classification-only; not present in detection args |
| `close_mosaic` | 10 | yes | disables mosaic for the **last 10 epochs** |

**Actually-active augmentations:** HSV color jitter, +/-10% translate, +/-50% scale,
50% horizontal flip, and full mosaic (turned off for the final 10 epochs via
`close_mosaic`). Everything geometric-heavy (rotation, shear, perspective, vertical
flip) and all the mixing augmentations (mixup / cutmix / copy_paste) are **off**.

**Worth flagging for camera-trap birds:** `mosaic=1.0` is aggressive — it stitches 4
frames into one, fragmenting the already-small birds and creating unnatural contexts
unlike the static-camera test footage; for tiny objects this is commonly dialed down.
`fliplr=0.5` is harmless and `flipud=0` is a sensible default (birds have a canonical
up/down orientation). But since **none of these were customized**, the augmentation
pipeline is identical across both models and is not the source of the RT-DETR-specific
behavior — the optimizer/LR mismatch in Section 2 is the far more likely culprit.

---

## 5. Where tunable values enter the project

The YAML `hyperparameters:` mapping is now passed through by both Ultralytics-backed
adapters (except `model_weights`, which initializes the model). This means settings such
as `optimizer`, `lr0`, `weight_decay`, and `mosaic` reach `model.train(...)` directly.
The adapters continue to own the generated data path and Ultralytics output path, and
RT-DETR continues to cap `workers` at 4 for the 8-CPU SLURM allocation.

Each run creates `work/dataset/images` as a symlink to the source images and writes its
own `work/dataset/labels` directory. Ultralytics therefore creates a run-local label
cache, avoiding races when many SLURM jobs prepare the same H23 split concurrently.

The sensitivity submission script writes persistent, generated YAML files with one
non-default value per run. RT-DETR's true-resume path remains intentionally different:
`resume=True` reloads optimizer, learning rate, and other training arguments from the
checkpoint, so new tuning values apply only to fresh runs.
