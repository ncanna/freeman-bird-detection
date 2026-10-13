"""Draw COCO annotations onto frames — a visual check of detection output.

Part of the H55 experiment tooling (see
``system_one_detector/h55_experiment/README.md``). Loads a COCO file, draws each
annotation's bbox with its score and System One attributes, optionally overlays
the 3x3 grid that defines the cell semantics, and writes per-frame PNGs plus a
montage contact sheet.

For the H55 System One runs the "bbox" is the predicted grid cell (a third of
the frame per side), so the grid overlay makes the model's answer legible.

Usage (from the repo root):

    python3 -m system_one_detector.tools.coco_draw \
        --coco outputs/h55/poc_h55_70tok_coco_t020.json \
        --images-dir data/H55/frames_sample5_stride10

Outputs land in ``<coco parent>/viz_<coco stem>/`` by default. Frames without
annotations are skipped unless ``--all`` is given (they would render as a bare
caption).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

GRID_COLOR = (190, 190, 190)
BOX_COLOR = (0, 140, 255)  # orange in BGR — readable on vegetation
TEXT_SHADOW = (0, 0, 0)
TEXT_COLOR = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _label(image: np.ndarray, text: str, origin: tuple[int, int], scale: float = 0.55) -> None:
    """White text with a black shadow so it survives any background."""
    x, y = origin
    cv2.putText(image, text, (x + 1, y + 1), FONT, scale, TEXT_SHADOW, 3, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), FONT, scale, TEXT_COLOR, 1, cv2.LINE_AA)


def draw_grid(image: np.ndarray) -> None:
    height, width = image.shape[:2]
    for col in (1, 2):
        x = round(width * col / 3)
        cv2.line(image, (x, 0), (x, height), GRID_COLOR, 1, cv2.LINE_AA)
    for row in (1, 2):
        y = round(height * row / 3)
        cv2.line(image, (0, y), (width, y), GRID_COLOR, 1, cv2.LINE_AA)


def draw_annotation(image: np.ndarray, ann: dict, categories: dict[int, str]) -> None:
    x, y, w, h = (float(v) for v in ann["bbox"])
    p1 = (round(x), round(y))
    p2 = (round(x + w), round(y + h))
    cv2.rectangle(image, p1, p2, BOX_COLOR, 2, cv2.LINE_AA)

    attrs = ann.get("attributes") or {}
    bits = [f"p(bird)={float(ann.get('score') or 0):.3f}"]
    if attrs.get("cell"):
        bits.append(f"cell={attrs['cell']} ({float(attrs.get('cell_probability') or 0):.2f})")
    if attrs.get("bird_count_label"):
        bits.append(f"count: {attrs['bird_count_label']}")
    name = categories.get(int(ann.get("category_id", 1)), "object")
    _label(image, f"{name} | {' · '.join(bits)}", (p1[0] + 4, max(18, p1[1] - 8)))


def render_frame(
    image_path: Path,
    annotations: list[dict],
    categories: dict[int, str],
    *,
    grid: bool = True,
) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise OSError(f"could not read {image_path}")
    if grid:
        draw_grid(image)
    for ann in annotations:
        draw_annotation(image, ann, categories)
    return image


def montage(tiles: list[np.ndarray], cols: int, tile_width: int) -> np.ndarray:
    """Contact sheet. Tiles keep their aspect ratio; empty cells stay white."""
    if not tiles:
        raise ValueError("montage needs at least one tile")
    resized = []
    for tile in tiles:
        scale = tile_width / tile.shape[1]
        size = (tile_width, round(tile.shape[0] * scale))
        resized.append(cv2.resize(tile, size, interpolation=cv2.INTER_AREA))
    tile_h = max(t.shape[0] for t in resized)
    rows = (len(resized) + cols - 1) // cols
    sheet = np.full((rows * (tile_h + 8) + 8, cols * (tile_width + 8) + 8, 3), 255, dtype=np.uint8)
    for index, tile in enumerate(resized):
        r, c = divmod(index, cols)
        y = 8 + r * (tile_h + 8)
        x = 8 + c * (tile_width + 8)
        sheet[y : y + tile.shape[0], x : x + tile.shape[1]] = tile
    return sheet


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--coco", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="default: <coco parent>/viz_<coco stem>")
    parser.add_argument("--no-grid", action="store_true", help="skip the 3x3 cell grid overlay")
    parser.add_argument("--all", action="store_true",
                        help="render every image entry, not only annotated ones")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--montage-cols", type=int, default=2)
    parser.add_argument("--tile-width", type=int, default=640)
    args = parser.parse_args(argv)

    coco = json.loads(args.coco.read_text())
    categories = {int(c["id"]): str(c.get("name") or c["id"]) for c in coco.get("categories", [])}
    anns_by_image: dict[int, list[dict]] = {}
    for ann in coco.get("annotations", []):
        anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)

    entries = list(coco.get("images", []))
    if not args.all:
        entries = [img for img in entries if int(img["id"]) in anns_by_image]
    if args.max_images is not None:
        entries = entries[: args.max_images]
    if not entries:
        print("nothing to draw (no annotated images; pass --all for bare captions)")
        return 0

    out_dir = args.out_dir or args.coco.parent / f"viz_{args.coco.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tiles = []
    skipped = 0
    for entry in entries:
        image_path = args.images_dir / entry["file_name"]
        try:
            rendered = render_frame(
                image_path, anns_by_image.get(int(entry["id"]), []), categories,
                grid=not args.no_grid,
            )
        except OSError as exc:
            print(f"  skipped {entry['file_name']}: {exc}")
            skipped += 1
            continue

        attrs = entry.get("attributes") or {}
        noul = attrs.get("noul_probability")
        header = f"{entry['file_name']}  p(bird)={float(noul):.3f}" if noul is not None else entry["file_name"]
        _label(rendered, header, (8, 22), scale=0.6)

        cv2.imwrite(str(out_dir / entry["file_name"]), rendered)
        tiles.append(rendered)
        print(f"  {entry['file_name']}  ({len(anns_by_image.get(int(entry['id']), []))} box(es))")

    if len(tiles) > 1:
        sheet = montage(tiles, cols=args.montage_cols, tile_width=args.tile_width)
        montage_path = out_dir / "montage.png"
        cv2.imwrite(str(montage_path), sheet)
        print(f"Montage → {montage_path}")
    print(f"Wrote {len(tiles)} annotated frame(s) → {out_dir}" + (f" ({skipped} skipped)" if skipped else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
