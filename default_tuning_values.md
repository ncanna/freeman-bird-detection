# RT-DETR & YOLO Training Settings — h23 "default" runs

All settings below are taken from each run's Ultralytics `work/runs/train/args.yaml`:

- **RT-DETR**: `outputs/rtdetr_h23_20260707_072734` (amp-on, 100 epochs, H200)
- **YOLO11**: `outputs/yolo11_h23_20260609_160816`

**Key point:** these runs use **Ultralytics defaults for essentially everything**. The
project YAML configs only set `model_weights`, `epochs`, `imgsz`, `batch`, `device`,
and `amp` — so RT-DETR and YOLO share **identical** settings. The only real differences
are the model weights and the dataloader `workers` (4 vs 8). Nothing in the augmentation
or optimizer space was actually tuned.

---

## 1. Core hyperparameters

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

| Parameter | Value (both) | Notes |
|---|---|---|
| `optimizer` | `auto` -> **SGD** | `auto` ignores the requested `lr0`/`momentum` and resolved to `SGD(lr=0.01, momentum=0.9)` for **both** models |
| `lr0` | 0.01 (effective) | initial learning rate |
| `lrf` | 0.01 | final LR = `lr0 * lrf` = 1e-4 |
| `cos_lr` | False | linear decay (not cosine) |
| `momentum` | 0.9 | |
| `weight_decay` | 0.0005 | |
| `warmup_epochs` | 3.0 | |
| `warmup_momentum` | 0.8 | |
| `warmup_bias_lr` | 0.1 | |
| `nbs` | 64 | nominal batch size for LR scaling |

**Comment (ties to the degradation + duplicate-box findings):** `optimizer=auto`
selected **SGD @ lr0=0.01** for both models (confirmed in the training logs). That is
the *intended* recipe for YOLO (a CNN), and YOLO trains fine. But **RT-DETR is a
DETR/transformer model**, and the official RT-DETR recipe uses **AdamW at ~1e-4**
(with an even lower backbone LR) and `weight_decay ~1e-4`. Training a transformer with
**SGD at 0.01 (~100x higher LR, wrong optimizer family)** is very consistent with the
observed behavior: per-epoch val mAP peaks at epoch 1 and then declines, `best.pt` ends
up being an early/near-pretrained checkpoint, and the object queries never sharpen to
one-per-bird (which is why the video shows many duplicate boxes). This is the most
likely single lever to change.

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

## 5. Where data augmentation lives in the project

**Short answer: nowhere yet.** A grep for augmentation keys (`mosaic`, `hsv`, `mixup`,
`fliplr`, `translate`, `degrees`, `copy_paste`, `erasing`, ...) across `hlwdetector/`
and `configs/` returns **no matches**. Every value in Section 4 comes straight from the
Ultralytics defaults because nothing in this project sets them.

The single control point is each adapter's `train()` method, where `model.train(...)` is
called with only a fixed set of arguments (data, epochs, imgsz, batch, device, amp,
workers, project, name) — no augmentation args:

- `hlwdetector/adapters/yolo_adapter.py` — `train()`; the `self._model.train(...)` calls
  (around lines 142 and 155).
- `hlwdetector/adapters/rtdetr_adapter.py` — `train()`; the `self._model.train(...)` call
  (around line 143). Note the **resume path** (line 163) uses `self._model.train(resume=True)`,
  which reloads all args from the checkpoint — so any augmentation kwargs would be ignored
  on a resumed run.

The config surface those methods read from:

- YAML `hyperparameters:` block (e.g. `configs/rtdetr_h23_full.yaml`,
  `configs/yolo11_h23_full.yaml`) -> `config.hyperparameters` -> read via `hp.get(...)`
  at the top of each `train()` (lines ~124-130).

So to actually tune augmentation you would (1) add keys such as `mosaic:`, `hsv_h:`,
`scale:` under `hyperparameters:` in the config, and (2) forward them from the adapter's
`train()` into the `model.train(...)` call. **Neither exists today**, which is why adding
augmentation keys to a config currently has no effect — the adapter never reads or passes
them.
