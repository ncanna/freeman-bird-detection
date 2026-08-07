"""Shared COCO detection metrics, computed with pycocotools COCOeval.

Adapters that do not get metrics from their training framework (swin, detr) route
through :func:`compute_coco_metrics` so every adapter reports the same quantities.

Callers pass detections in COCO result format and are responsible for two
conversions the evaluator cannot do for them:

* box format — model output is xyxy, COCO results are ``[x, y, w, h]``
* class space — a model's contiguous class index back to the dataset's
  ``category_id`` (use ``category_ids(split_view)``)

Detections should be collected at a *low* score threshold (see
``EVAL_SCORE_THRESHOLD``), not the threshold used for visualization. AP is the
area under the full precision/recall curve; pre-filtering at e.g. 0.5 truncates
the curve and systematically understates AP.
"""

from __future__ import annotations

import contextlib
import io
import logging
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from hlwdetector.results import MetricsDict

if TYPE_CHECKING:
    from hlwdetector.dataset_manager import SplitView

logger = logging.getLogger(__name__)

# Default score threshold for detections fed to COCOeval. Low enough to keep the
# tail of the PR curve, high enough to keep the detection list manageable.
EVAL_SCORE_THRESHOLD = 0.001

def _stat_keys(max_dets_thresholds: list[int]) -> tuple[str, ...]:
    """Labels for the 12 scalars COCOeval.summarize() writes to `stats`, in order.

    The three AR entries are reported at each of ``params.maxDets``, so their
    names follow whatever the caller asked for rather than assuming [1, 10, 100].
    """
    low, mid, high = max_dets_thresholds
    return (
        "AP", "AP50", "AP75", "APs", "APm", "APl",
        f"AR{low}", f"AR{mid}", f"AR{high}", "ARs", "ARm", "ARl",
    )

# Indices into COCOeval's accumulated `precision` array [T, R, K, A, M]:
# T=IoU threshold (0 -> 0.50), A=area range (0 -> "all"), M=maxDets (2 -> the
# largest entry of params.maxDets).
_IOU_50 = 0
_AREA_ALL = 0
_MAXDETS_LARGEST = 2


def category_ids(split_view: "SplitView") -> list[str]:
    """COCO category ids ordered so that index == the model's class index.

    Sorting by ``id`` matches how ``visualization/pipeline.py`` assigns ground
    truth class indices, so predictions and GT stay in the same class space.
    """
    return [cat["id"] for cat in sorted(split_view.categories, key=lambda c: c["id"])]


def _empty_metrics(note: str) -> MetricsDict:
    logger.warning("COCO evaluation produced no usable results: %s", note)
    return MetricsDict(
        precision=0.0, recall=0.0, f1=0.0, map50=0.0, map50_95=0.0, raw={"note": note}
    )


def _build_coco_gt(split_view: "SplitView") -> COCO:
    """Build an in-memory COCO ground truth object for one split.

    Constructed from the SplitView rather than ``coco_json`` on disk because a
    split is a subset of that file — evaluating against the whole file would
    count every other split's images as missed detections.
    """
    annotations = []
    for i, ann in enumerate(split_view.annotations):
        ann = dict(ann)
        # COCO.createIndex and COCOeval hard-require these three.
        ann.setdefault("id", i + 1)
        ann.setdefault("iscrowd", 0)
        if "area" not in ann:
            _, _, w, h = ann["bbox"]
            ann["area"] = float(w) * float(h)
        annotations.append(ann)

    coco_gt = COCO()
    coco_gt.dataset = {
        "images": split_view.images,
        "annotations": annotations,
        "categories": split_view.categories,
    }
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt.createIndex()
    return coco_gt


def _precision_recall_at_best_f1(evaluator: COCOeval) -> tuple[float, float, float]:
    """Scalar precision/recall from the IoU=0.50 PR curve, at its best-F1 point.

    COCOeval reports AP but no single operating point. Taking the best-F1 point
    on the curve matches the semantics of Ultralytics' ``box.mp``/``box.mr``, so
    the numbers stay comparable to what the YOLO and RT-DETR adapters report.
    """
    # [T, R, K, A, M] -> mean over classes at IoU 0.50, area "all", largest maxDets.
    precision_curve = evaluator.eval["precision"][
        _IOU_50, :, :, _AREA_ALL, _MAXDETS_LARGEST
    ]
    # -1 marks recall points COCOeval could not evaluate; drop them before averaging.
    precision_curve = np.where(precision_curve > -1, precision_curve, np.nan)
    with np.errstate(invalid="ignore"):
        precision = np.nanmean(precision_curve, axis=1)

    recall = np.asarray(evaluator.params.recThrs, dtype=float)
    valid = ~np.isnan(precision)
    if not valid.any():
        return 0.0, 0.0, 0.0

    p, r = precision[valid], recall[valid]
    f1 = 2 * p * r / np.maximum(p + r, 1e-9)
    best = int(np.argmax(f1))
    return float(p[best]), float(r[best]), float(f1[best])


def compute_coco_metrics(
    split_view: "SplitView",
    detections: list[dict],
    max_dets: int = 100,
) -> MetricsDict:
    """Evaluate COCO-format detections against a split and return a MetricsDict.

    Args:
        split_view: the split being evaluated; supplies the ground truth.
        detections: COCO results — dicts with ``image_id``, ``category_id``,
            ``bbox`` as ``[x, y, w, h]``, and ``score``. Collect these at a low
            score threshold; see the module docstring.
        max_dets: detections per image considered by COCOeval.

    Returns:
        ``map50_95`` is AP@[.50:.95] and ``map50`` is AP@0.50, both averaged over
        classes. ``precision``/``recall``/``f1`` are the best-F1 point on the
        IoU=0.50 curve. ``raw`` carries all 12 COCO summary statistics; note that
        h23 birds (median √area ~56px, so area ~3.1k) fall in COCO's *medium*
        bucket (32²–96²), which makes ``APm`` the size band to watch — ``APs``
        reports -1 because the split contains no small objects at all.
    """
    if not detections:
        return _empty_metrics("no detections above the evaluation score threshold")
    if not split_view.annotations:
        return _empty_metrics(f"split '{split_view.split}' has no ground truth annotations")

    coco_gt = _build_coco_gt(split_view)

    # pycocotools prints progress on every call; it would flood experiment.log
    # once per epoch. Capture it and emit one summary line instead.
    with contextlib.redirect_stdout(io.StringIO()):
        # loadRes mutates the dicts it is given (it adds "id" and "area").
        coco_dt = coco_gt.loadRes(deepcopy(detections))
        evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
        evaluator.params.maxDets = [1, 10, max_dets]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    stats = {
        key: float(value)
        for key, value in zip(_stat_keys(evaluator.params.maxDets), evaluator.stats)
    }
    map50_95, map50 = stats["AP"], stats["AP50"]
    precision, recall, f1 = _precision_recall_at_best_f1(evaluator)

    logger.info(
        "COCOeval on '%s' (%d images, %d GT, %d detections): "
        "mAP50-95=%.4f mAP50=%.4f APm=%.4f P=%.4f R=%.4f F1=%.4f",
        split_view.split,
        len(split_view.images),
        len(split_view.annotations),
        len(detections),
        map50_95,
        map50,
        stats["APm"],
        precision,
        recall,
        f1,
    )

    return MetricsDict(
        precision=precision,
        recall=recall,
        f1=f1,
        map50=map50,
        map50_95=map50_95,
        raw=stats,
    )
