"""Closed questions a System One model can answer about a camera-trap frame.

Gold labels come from COCO boxes. The text state passed to the model must
not include these labels.
"""

from __future__ import annotations

from typing import Any


COUNT_CRITERIA = [
    "no birds are visible",
    "exactly one bird is visible",
    "a few birds are visible (2 or 3)",
    "many birds are visible (4 or more)",
]

# 3x3 cells plus an explicit empty option. Ten options stays under the
# ~20-option quality drop reported for Laya and Jev-Omni.
CELL_NAMES = ("tl", "tc", "tr", "ml", "mc", "mr", "bl", "bc", "br", "none")
CELL_CRITERIA = {
    "tl": "the largest bird's center is in the top-left third of the frame",
    "tc": "the largest bird's center is in the top-center third of the frame",
    "tr": "the largest bird's center is in the top-right third of the frame",
    "ml": "the largest bird's center is in the middle-left third of the frame",
    "mc": "the largest bird's center is in the center third of the frame",
    "mr": "the largest bird's center is in the middle-right third of the frame",
    "bl": "the largest bird's center is in the bottom-left third of the frame",
    "bc": "the largest bird's center is in the bottom-center third of the frame",
    "br": "the largest bird's center is in the bottom-right third of the frame",
    "none": "no bird is visible",
}

_ROW = ("t", "m", "b")
_COL = ("l", "c", "r")

DEFAULT_QUESTION_IDS = (
    "bird_present",
    "bird_count",
    "largest_bird_cell",
    "bird_behavior",
)


def count_level(n_birds: int) -> int:
    if n_birds <= 0:
        return 0
    if n_birds == 1:
        return 1
    if n_birds <= 3:
        return 2
    return 3


def cell_of_box(bbox: list[float], width: float, height: float) -> str:
    """Map a COCO ``[x, y, w, h]`` box to a 3x3 cell from its center."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive to place a box in a cell")
    x, y, w, h = bbox
    cx = min(max(x + w / 2.0, 0.0), width - 1e-6)
    cy = min(max(y + h / 2.0, 0.0), height - 1e-6)
    col = min(2, int(cx / width * 3))
    row = min(2, int(cy / height * 3))
    return _ROW[row] + _COL[col]


def _largest_box(annotations: list[dict]) -> dict | None:
    boxes = [ann for ann in annotations if ann.get("bbox")]
    if not boxes:
        return None
    return max(boxes, key=lambda ann: float(ann["bbox"][2]) * float(ann["bbox"][3]))


def build_questions(
    question_ids: list[str] | None = None,
    *,
    categories: list[dict] | None = None,
    extra: dict[str, dict] | None = None,
) -> dict[str, dict]:
    """Build the question map for one request.

    ``primary_category`` is included only when COCO categories are supplied
    and the option count stays at or below 20 (``none`` and ``other`` included).
    """
    ids = list(question_ids or DEFAULT_QUESTION_IDS)
    bank: dict[str, dict] = {
        "bird_present": {
            "type": "noul",
            "instructions": "Is at least one bird visible in this camera-trap frame?",
            "criteria": {
                "true": "one or more birds are visible",
                "false": "no bird is visible",
            },
        },
        "bird_count": {
            "type": "score",
            "instructions": "How many birds are visible in this camera-trap frame?",
            "criteria": list(COUNT_CRITERIA),
        },
        "largest_bird_cell": {
            "type": "choice",
            "instructions": "Where is the center of the largest visible bird?",
            "criteria": dict(CELL_CRITERIA),
        },
        "bird_behavior": {
            "type": "choice",
            "instructions": (
                "What is the most clearly visible behavior of the most prominent bird? "
                "Choose unclear if the image does not provide enough evidence."
            ),
            "criteria": {
                "foraging": "searching for or handling food, including probing or gleaning",
                "perching_or_resting": "stationary on a perch, standing, or resting",
                "flying": "in flight with the body airborne",
                "walking_or_hopping": "moving on a surface by walking or hopping",
                "calling": "visibly producing a call or song",
                "interacting": "interacting with another animal",
                "other": "another visible behavior not listed",
                "unclear": "a bird is visible but its behavior cannot be determined",
                "none": "no bird is visible",
            },
        },
    }
    if "primary_category" in ids:
        names = [str(cat.get("name") or cat.get("id")) for cat in (categories or [])]
        options = ["none", *names, "other"]
        if not names:
            raise ValueError("primary_category requires COCO categories")
        if len(options) > 20:
            raise ValueError(
                f"primary_category has {len(options)} options. Laya and Jev-Omni "
                "drop off above about 20. Split the label set or drop this question."
            )
        bank["primary_category"] = {
            "type": "choice",
            "instructions": "What is the category of the largest visible bird?",
            "criteria": {
                name: ("no bird is visible" if name == "none" else "a bird of this category, or an unlisted category" if name == "other" else f"the largest bird is a {name}")
                for name in options
            },
        }
    unknown = [qid for qid in ids if qid not in bank and qid not in (extra or {})]
    if unknown:
        raise ValueError(f"unknown question ids: {unknown}. Known: {sorted(bank)}")
    selected = {qid: bank[qid] for qid in ids if qid in bank}
    if extra:
        selected.update(extra)
    return selected


def labels_for_frame(
    question_ids: list[str],
    annotations: list[dict],
    *,
    width: float | None,
    height: float | None,
    category_names: dict[int, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Gold labels for the questions this frame can answer.

    ``largest_bird_cell`` is omitted when image size is unknown, so metrics
    skip it instead of inventing a cell.
    """
    n_birds = len(annotations)
    labels: dict[str, dict[str, Any]] = {}
    if "bird_present" in question_ids:
        labels["bird_present"] = {"type": "noul", "noul": 1.0 if n_birds else 0.0}
    if "bird_count" in question_ids:
        level = count_level(n_birds)
        labels["bird_count"] = {"type": "score", "score": float(level), "level": level}
    if "largest_bird_cell" in question_ids:
        if n_birds == 0:
            labels["largest_bird_cell"] = {"type": "choice", "choice": "none"}
        elif width and height:
            box = _largest_box(annotations)
            if box is not None:
                labels["largest_bird_cell"] = {
                    "type": "choice",
                    "choice": cell_of_box(box["bbox"], width, height),
                }
    if "primary_category" in question_ids and category_names is not None:
        if n_birds == 0:
            choice = "none"
        else:
            box = _largest_box(annotations)
            cat_id = int(box["category_id"]) if box and box.get("category_id") is not None else None
            name = category_names.get(cat_id) if cat_id is not None else None
            choice = name or "other"
        labels["primary_category"] = {"type": "choice", "choice": choice}
    return labels


def text_state(file_name: str, video_stem: str, frame_index: int | None) -> dict[str, Any]:
    """Text state for a frame. Contains no annotation-derived facts."""
    state: dict[str, Any] = {
        "source": "freeman camera trap",
        "video_stem": video_stem,
        "frame_name": file_name,
        "note": (
            "This text does not describe the picture. "
            "Use the attached image or video when the backend provides one."
        ),
    }
    if frame_index is not None:
        state["frame_index"] = frame_index
    return state
