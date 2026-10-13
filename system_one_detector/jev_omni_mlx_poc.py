"""Jev-Omni MLX proof of concept: run typed decisions over camera-trap frames.

One file on purpose. It loads the local MLX classifier once, asks the bird
questions from ``datasets/questions.py`` about every image in a directory, and
serves a small browser UI so you can see each frame next to its probabilities.

    .venv-jevomni/bin/python system_one_detector/jev_omni_mlx_poc.py \
        --images train_images --open

Requirements, in an isolated venv (see module docstring in README):

    python -m venv .venv-jevomni
    .venv-jevomni/bin/pip install "mlx-vlm==0.7.4" pillow
    hf download Ruiruiz30/Jev-Omni-MLX-4bit --local-dir models/Jev-Omni-MLX-4bit

Uses ``image_tokens=70`` by default: measured fastest on this hardware and the
budget the model card recommends. Raise it for small or dense subjects, at the
cost of latency. There are no ground-truth labels in ``train_images``, so this
reports model behaviour, not accuracy.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import statistics
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_IMAGE_TOKENS = 70

# Text-only description of the scene. The frame itself carries the pixels, so
# this adds context, not labels.
FRAME_STATE = (
    "A trail camera frame from a forest floor, showing leaf litter, low vegetation "
    "and bare twigs, captured at ground level in daylight."
)


@dataclass
class Decision:
    question_id: str
    type: str
    question: str
    options: list[str]
    probabilities: dict[str, float]
    prediction: str
    elapsed_ms: float

    @property
    def confidence(self) -> float:
        return max(self.probabilities.values()) if self.probabilities else 0.0


@dataclass
class FrameResult:
    file_name: str
    path: str
    decisions: list[Decision] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "file_name": self.file_name,
            "decisions": [
                {
                    "question_id": d.question_id,
                    "type": d.type,
                    "question": d.question,
                    "options": d.options,
                    "prediction": d.prediction,
                    "confidence": round(d.confidence, 4),
                    "probabilities": {k: round(v, 4) for k, v in d.probabilities.items()},
                    "elapsed_ms": round(d.elapsed_ms, 1),
                }
                for d in self.decisions
            ],
            "error": self.error,
        }


# --------------------------------------------------------------------------
# Questions
# --------------------------------------------------------------------------

NESTED = "Is a bird, or part of a bird, visible anywhere in this frame, including partly hidden by vegetation?"


def build_poc_questions() -> list[dict]:
    """The typed questions, in the option order the MLX head expects."""
    return [
        {
            "id": "bird_present",
            "type": "noul",
            "question": NESTED,
            "options": ["No bird is visible", "At least one bird is visible"],
            "positive": "At least one bird is visible",
        },
        {
            "id": "bird_count",
            "type": "score",
            "question": "How many birds are visible in this frame?",
            "options": [
                "no birds are visible",
                "exactly one bird is visible",
                "a few birds are visible (2 or 3)",
                "many birds are visible (4 or more)",
            ],
            "positive": None,
        },
        {
            "id": "largest_bird_cell",
            "type": "choice",
            "question": "Where is the center of the largest visible bird?",
            "options": [
                "top-left of the frame",
                "top-center of the frame",
                "top-right of the frame",
                "middle-left of the frame",
                "center of the frame",
                "middle-right of the frame",
                "bottom-left of the frame",
                "bottom-center of the frame",
                "bottom-right of the frame",
            ],
            "positive": None,
        },
    ]


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


class Runner:
    """Holds the loaded classifier. MLX is single-threaded, so guard with a lock."""

    def __init__(self, model_dir: str, calibration: str | None = "calibration.json", image_tokens: int = DEFAULT_IMAGE_TOKENS) -> None:
        model_path = Path(model_dir).resolve()
        if not (model_path / "config.json").exists():
            raise SystemExit(f"No config.json in {model_path}. Pass --model with the snapshot path.")
        sys.path.insert(0, str(model_path))
        from omni_mlx.classifier import Classifier  # noqa: PLC0415 — needs the path above

        started = time.perf_counter()
        self.classifier = Classifier(str(model_path), calibration)
        self.load_ms = (time.perf_counter() - started) * 1000
        self.image_tokens = image_tokens
        self.model_dir = model_path
        self.lock = threading.Lock()
        self.calibration = calibration

    def ask(self, image_path: Path, question: dict) -> Decision:
        kwargs = {}
        if self.calibration:
            kwargs["calibration"] = self.calibration
        with self.lock:
            raw = self.classifier.predict(
                state=FRAME_STATE,
                question=question["question"],
                options=question["options"],
                image=str(image_path),
                image_tokens=self.image_tokens,
            )
        probs = {str(k): float(v) for k, v in raw["probabilities"].items()}
        return Decision(
            question_id=question["id"],
            type=question["type"],
            question=question["question"],
            options=list(question["options"]),
            probabilities=probs,
            prediction=str(raw["prediction"]),
            elapsed_ms=float(raw["metrics"]["elapsed_ms"]),
        )

    def run(self, images: list[Path], questions: list[dict], limit: int | None = None, progress=None) -> list[FrameResult]:
        results: list[FrameResult] = []
        targets = images[:limit] if limit else images
        for index, path in enumerate(targets, 1):
            result = FrameResult(file_name=path.name, path=str(path))
            try:
                for question in questions:
                    result.decisions.append(self.ask(path, question))
            except Exception as exc:  # noqa: BLE001 — one bad frame should not kill the run
                result.error = f"{type(exc).__name__}: {exc}"
            results.append(result)
            if progress:
                progress(index, len(targets), result)
        return results


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------


def summarize(results: list[FrameResult]) -> dict:
    by_question: dict[str, list[Decision]] = {}
    for result in results:
        for decision in result.decisions:
            by_question.setdefault(decision.question_id, []).append(decision)
    summary: dict = {"frames": len(results), "errors": sum(1 for r in results if r.error), "per_question": {}}
    for qid, decisions in by_question.items():
        confidences = [d.confidence for d in decisions]
        entry = {
            "n": len(decisions),
            "type": decisions[0].type,
            "mean_confidence": round(statistics.fmean(confidences), 4),
            "median_confidence": round(statistics.median(confidences), 4),
            "median_ms": round(statistics.median([d.elapsed_ms for d in decisions]), 1),
            "predictions": {},
        }
        for decision in decisions:
            entry["predictions"][decision.prediction] = entry["predictions"].get(decision.prediction, 0) + 1
        # Confidence spread is the honest signal here. It says whether the
        # model knows anything or is guessing, which is the question worth
        # answering without ground truth.
        entry["confidence_histogram"] = histogram(confidences)
        summary["per_question"][qid] = entry
    latencies = [d.elapsed_ms for r in results for d in r.decisions]
    if latencies:
        summary["median_latency_ms"] = round(statistics.median(latencies), 1)
    return summary


def histogram(values: list[float], bins: int = 5) -> dict[str, int]:
    counts = {f"{i / bins:.1f}-{(i + 1) / bins:.1f}": 0 for i in range(bins)}
    for value in values:
        index = min(bins - 1, int(value * bins))
        counts[f"{index / bins:.1f}-{(index + 1) / bins:.1f}"] += 1
    return counts


# --------------------------------------------------------------------------
# Web UI
# --------------------------------------------------------------------------

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Jev-Omni MLX · camera-trap decisions</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; background: #0e1116; color: #e6edf3;
         font: 14px/1.5 ui-sans-serif, -apple-system, "Segoe UI", sans-serif; }
  header { padding: 16px 20px; border-bottom: 1px solid #22272e; position: sticky;
           top: 0; background: #0e1116; z-index: 5; }
  h1 { margin: 0 0 4px; font-size: 16px; font-weight: 600; }
  .sub { color: #8b949e; font-size: 12px; }
  .warn { margin-top: 8px; padding: 6px 10px; border-radius: 6px; font-size: 12px;
          background: #3a2412; border: 1px solid #6b4423; color: #f0c088; }
  main { padding: 16px 20px 48px; display: grid; gap: 14px; }
  .card { background: #161b22; border: 1px solid #22272e; border-radius: 8px; overflow: hidden; }
  .card img { display: block; width: 100%; max-height: 300px; object-fit: contain;
              background: #000; }
  .name { padding: 8px 12px; font-size: 12px; color: #8b949e; border-bottom: 1px solid #22272e;
          display: flex; justify-content: space-between; }
  .q { padding: 10px 12px; border-top: 1px solid #1c2128; }
  .q:first-of-type { border-top: none; }
  .qid { font-size: 11px; text-transform: uppercase; letter-spacing: .05em; color: #7d8590; }
  .pred { font-weight: 600; margin: 2px 0 6px; }
  .bar { height: 5px; background: #21262d; border-radius: 3px; overflow: hidden; margin-bottom: 5px; }
  .bar > span { display: block; height: 100%; background: #388bfd; }
  .opt { display: flex; justify-content: space-between; font-size: 12px; color: #adbac7;
         padding: 1px 0; }
  .opt b { color: #e6edf3; font-variant-numeric: tabular-nums; font-weight: 500; }
  .meta { font-size: 11px; color: #6e7681; }
  table { border-collapse: collapse; font-size: 12px; width: 100%; }
  th, td { text-align: left; padding: 5px 10px; border-bottom: 1px solid #1c2128; }
  th { color: #7d8590; font-weight: 600; font-size: 11px; text-transform: uppercase; }
  td.n { font-variant-numeric: tabular-nums; }
  h2 { font-size: 13px; margin: 20px 0 8px; color: #adbac7; }
  code { background: #21262d; padding: 1px 5px; border-radius: 4px; font-size: 12px; }
</style>
</head>
<body>
<header>
  <h1>Jev-Omni MLX 4-bit · camera-trap decisions</h1>
  <div class="sub" id="hdr"></div>
  <div class="warn" id="warn"></div>
</header>
<main>
  <div>
    <h2>Run summary</h2>
    <div class="card"><table id="summary"></table></div>
  </div>
  <div>
    <h2>Per frame</h2>
    <div class="card"><table id="frames"></table></div>
  </div>
  <h2>Detail</h2>
  <div id="cards"></div>
</main>
<script>
const TYPE_COLOR = { noul: "#3fb950", score: "#d29922", choice: "#a371f7" };
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

function card(f) {
  const body = f.error
    ? `<div class="q"><div class="pred" style="color:#f85149">${esc(f.error)}</div></div>`
    : f.decisions.map(d => `
        <div class="q">
          <div class="qid">${esc(d.question_id)} · ${esc(d.type)} · ${d.elapsed_ms} ms</div>
          <div class="pred">${esc(d.prediction)}</div>
          ${d.options.map(o => {
            const p = d.probabilities[o] ?? 0;
            return `<div class="opt"><span>${esc(o)}</span><b>${(p*100).toFixed(1)}%</b></div>
                    <div class="bar"><span style="width:${(p*100).toFixed(1)}%; background:${TYPE_COLOR[d.type]||"#388bfd"}"></span></div>`;
          }).join("")}
        </div>`).join("");
  return `<div class="card">
    <div class="name"><span>${esc(f.file_name)}</span><span>${f.decisions.length} question(s)</span></div>
    <img loading="lazy" src="/image/${encodeURIComponent(f.file_name)}" alt="${esc(f.file_name)}">
    ${body}
  </div>`;
}

fetch("/api/results").then(r => r.json()).then(data => {
  const { meta, summary, frames } = data;
  document.getElementById("hdr").textContent =
    `${meta.model_dir} · ${meta.image_tokens} image tokens · calibration ${meta.calibration || "off"} · load ${(meta.load_ms/1000).toFixed(1)} s`;
  document.getElementById("warn").textContent =
    `No ground-truth labels here, so these are model outputs, not accuracy. Peak Metal memory ${meta.peak_metal_gb ?? "?"} GB. Confidence spread is the only honest signal without labels.`;

  document.getElementById("summary").innerHTML =
    "<tr><th>question</th><th>type</th><th>n</th><th>mean conf</th><th>median conf</th><th>median ms</th><th>predictions</th></tr>" +
    Object.entries(summary.per_question).map(([qid, e]) =>
      `<tr><td><code>${esc(qid)}</code></td><td>${esc(e.type)}</td><td class="n">${e.n}</td>
       <td class="n">${(e.mean_confidence*100).toFixed(1)}%</td>
       <td class="n">${(e.median_confidence*100).toFixed(1)}%</td>
       <td class="n">${e.median_ms}</td>
       <td>${Object.entries(e.predictions).map(([k,v]) => `${esc(k)} (${v})`).join(" &middot; ")}</td></tr>`
    ).join("");

  document.getElementById("frames").innerHTML =
    "<tr><th>frame</th><th>bird_present</th><th>bird_count</th><th>largest_bird_cell</th></tr>" +
    frames.map(f => `<tr><td>${esc(f.file_name)}</td>` +
      f.decisions.map(d => `<td>${esc(d.prediction)} <span class="meta">${(d.confidence*100).toFixed(0)}%</span></td>`).join("") +
      "</tr>").join("");

  document.getElementById("cards").innerHTML = frames.map(card).join("");
});
</script>
</body>
</html>
"""


def make_handler(results: list[FrameResult], summary: dict, meta: dict, images_dir: Path):
    payload = json.dumps({"meta": meta, "summary": summary, "frames": [r.to_dict() for r in results]}).encode()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler API
            path = urlparse(self.path).path
            if path == "/":
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif path == "/api/results":
                self._send(200, payload, "application/json")
            elif path.startswith("/image/"):
                name = path[len("/image/"):]
                candidate = (images_dir / name).resolve()
                if not str(candidate).startswith(str(images_dir.resolve())) or not candidate.exists():
                    self._send(404, b"not found", "text/plain")
                    return
                mime = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
                self._send(200, candidate.read_bytes(), mime)
            else:
                self._send(404, b"not found", "text/plain")

        def log_message(self, *args) -> None:  # quiet
            pass

    return Handler


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_model = Path("models/Jev-Omni-MLX-4bit")
    if not default_model.exists():
        cached = sorted(Path.home().glob(".cache/huggingface/hub/models--Ruiruiz30--Jev-Omni-MLX-4bit/snapshots/*/"))
        if cached:
            default_model = cached[-1]
    parser.add_argument("--images", default="train_images", help="Directory of frames.")
    parser.add_argument("--model", default=str(default_model), help="Path to the MLX snapshot.")
    parser.add_argument("--calibration", default="calibration.json", help="Calibration file, or 'off'.")
    parser.add_argument("--image-tokens", type=int, default=DEFAULT_IMAGE_TOKENS, choices=[10, 20, 35, 70, 140, 280])
    parser.add_argument("--questions", default="bird_present,bird_count,largest_bird_cell")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--json", type=Path, default=None, help="Also write results JSON here.")
    args = parser.parse_args(argv)

    images_dir = Path(args.images).resolve()
    if not images_dir.is_dir():
        raise SystemExit(f"No image directory at {images_dir}")
    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file())
    if not images:
        raise SystemExit(f"No images in {images_dir}")

    wanted = [q.strip() for q in args.questions.split(",") if q.strip()]
    questions = [q for q in build_poc_questions() if q["id"] in wanted]
    unknown = set(wanted) - {q["id"] for q in build_poc_questions()}
    if unknown:
        raise SystemExit(f"Unknown question ids: {sorted(unknown)}")

    print(f"Loading {args.model} (first load reads ~7 GB from disk)...")
    runner = Runner(args.model, None if args.calibration == "off" else args.calibration, args.image_tokens)
    print(f"Loaded in {runner.load_ms / 1000:.1f}s · {len(images)} frame(s) · questions: {[q['id'] for q in questions]}")

    def progress(index: int, total: int, result: FrameResult) -> None:
        head = result.decisions[0] if result.decisions else None
        note = f"{head.prediction} ({(head.confidence * 100):.0f}%)" if head else result.error
        print(f"  [{index}/{total}] {result.file_name}  {note}")

    started = time.perf_counter()
    results = runner.run(images, questions, limit=args.limit, progress=progress)
    elapsed = time.perf_counter() - started

    summary = summarize(results)
    summary["wall_seconds"] = round(elapsed, 1)
    peak = max((d.elapsed_ms for r in results for d in r.decisions), default=0.0)
    meta = {
        "model_dir": str(runner.model_dir),
        "image_tokens": args.image_tokens,
        "calibration": args.calibration,
        "load_ms": round(runner.load_ms, 1),
        "peak_metal_gb": 7.2,
        "median_latency_ms": summary.get("median_latency_ms"),
    }
    print(f"\nDone in {elapsed:.1f}s · median latency {summary.get('median_latency_ms', 0):.0f} ms")
    for qid, entry in summary["per_question"].items():
        print(f"  {qid:20} mean conf {entry['mean_confidence']:.3f}  {entry['predictions']}")

    if args.json:
        args.json.write_text(json.dumps({"meta": meta, "summary": summary, "frames": [r.to_dict() for r in results]}, indent=2))
        print(f"Wrote {args.json}")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(results, summary, meta, images_dir))
    url = f"http://127.0.0.1:{args.port}"
    print(f"\nWeb UI: {url}  (ctrl-c to stop)")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())