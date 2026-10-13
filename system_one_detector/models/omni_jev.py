"""OmniJev v1.1 from tinnel123. A 4B Qwen3.5 LoRA with decision heads.

This is not OmniJev/OneJev-27B. This is a separate release from
tinnel123666888, published 2026-09-26, holding a LoRA adapter plus decision and
ordinal heads. The Qwen backbone is not bundled.

    git clone https://github.com/tinnel123666888/OmniJev && cd OmniJev
    pip install -r requirements.txt
    hf download tinnel123/OmniJev --revision v1.1 --local-dir ckpt
    hf download Qwen/Qwen3.5-4B --local-dir base

Then point ``adapter_dir`` at ``ckpt`` and ``base_model_dir`` at ``base``.

v1.1 reads still images as a numbered panel and applies the checkpoint's noul
bias before temperature scaling. That bias moves the yes/no boundary: the 4B
checkpoint carries ``biases.noul=0.0531``, so a raw 0.5 threshold on the
released head is not the serving path. Set ``noul_bias`` from
``release_manifest.json`` if you want the corrected read, and expect
calibration on your own data either way.

Its reported macro accuracy is over its own 30 task families, several of which
are games and GUI actions. That is not a camera-trap number.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from system_one_detector.models.base import DecisionBackend
from system_one_detector.models.registry import register_backend
from system_one_detector.util.schema import DecisionResult, parse_answer, validate_questions

logger = logging.getLogger(__name__)


def read_noul_bias(adapter_dir: str | None, override: float | None) -> float | None:
    """Return the checkpoint's noul bias, if one is recorded."""
    if override is not None:
        return override
    if not adapter_dir:
        return None
    manifest = Path(adapter_dir) / "release_manifest.json"
    if not manifest.exists():
        return None
    try:
        data = json.loads(manifest.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("could not read %s: %s", manifest, exc)
        return None
    biases = data.get("biases") if isinstance(data, dict) else None
    if isinstance(biases, dict) and "noul" in biases:
        try:
            return float(biases["noul"])
        except (TypeError, ValueError):
            return None
    return None


@register_backend("omni_jev")
class OmniJevBackend(DecisionBackend):
    name = "omni_jev"
    sees_pixels = True

    def __init__(
        self,
        *,
        adapter_dir: str | None = None,
        base_model_dir: str | None = None,
        noul_bias: float | None = None,
        no_panels: bool = False,
        device: str | None = None,
    ) -> None:
        self.adapter_dir = adapter_dir
        self.base_model_dir = base_model_dir
        self.noul_bias_override = noul_bias
        self.no_panels = no_panels
        self.device = device
        self._model = None
        self._bias_applied = False

    @classmethod
    def from_config(cls, config) -> "OmniJevBackend":
        return cls(
            adapter_dir=config.adapter_dir,
            base_model_dir=config.base_model_dir,
            noul_bias=config.noul_bias,
            no_panels=config.no_panels,
            device=config.device,
        )

    def _load(self):
        if self._model is not None:
            return self._model
        if not self.adapter_dir or not self.base_model_dir:
            raise RuntimeError(
                "omni_jev needs adapter_dir and base_model_dir. Clone tinnel123666888/OmniJev, "
                "pip install -r requirements.txt, then hf download tinnel123/OmniJev "
                "--revision v1.1 --local-dir ckpt and Qwen/Qwen3.5-4B --local-dir base."
            )
        for label, path in (("adapter_dir", self.adapter_dir), ("base_model_dir", self.base_model_dir)):
            if not Path(path).exists():
                raise FileNotFoundError(f"{label} not found: {path}")
        try:
            from mso.infer import MSO1
        except ImportError as exc:
            raise RuntimeError(
                "mso is not importable. Add the cloned OmniJev repo to sys.path or install its "
                "requirements. See the module docstring."
            ) from exc
        import sys

        repo = str(Path(self.adapter_dir).resolve().parent)
        if repo not in sys.path:
            sys.path.insert(0, repo)
        if self.no_panels:
            os_env_backup = __import__("os").environ.get("MSO_NO_PANELS")
            __import__("os").environ["MSO_NO_PANELS"] = "1"
        logger.info("Loading OmniJev adapter %s over base %s", self.adapter_dir, self.base_model_dir)
        self._model = MSO1(self.adapter_dir, self.base_model_dir)
        if self.no_panels:
            __import__("os").environ["MSO_NO_PANELS"] = os_env_backup or ""
        return self._model

    def decide(self, state, questions, *, media_path=None, modality="text") -> DecisionResult:
        questions = validate_questions(questions)
        if modality != "text" and not media_path:
            raise ValueError(f"modality {modality!r} requires media_path")
        if modality == "audio":
            raise ValueError("OmniJev takes images and video mosaics, not standalone audio.")
        model = self._load()
        images: list[str] = []
        if modality in {"image", "video"}:
            images.append(media_path)
        if not images:
            logger.warning("omni_jev got no media. It needs at least one image to answer about it.")
        raw_state: dict = {"images": images}
        if isinstance(state, dict):
            for key, value in state.items():
                if key != "images":
                    raw_state[key] = value
        raw = model.system_one(raw_state, questions)
        if not isinstance(raw, dict):
            raise RuntimeError(f"OmniJev returned {type(raw).__name__}, expected a dict of answers")
        answers = {}
        for qid, value in raw.items():
            question = questions.get(qid)
            if question is None:
                continue
            answer = value if hasattr(value, "noul") and not isinstance(value, dict) else parse_answer(_as_dict(value, question))
            if not self._bias_applied and question["type"] == "noul" and answer.noul is not None:
                bias = read_noul_bias(self.adapter_dir, self.noul_bias_override)
                if bias:
                    answer.noul = min(1.0, max(0.0, answer.noul + bias))
                    self._bias_applied = True
            answers[qid] = answer
        return DecisionResult(model="omni_jev-v1.1", answers=answers, raw={"raw": {k: _jsonable(v) for k, v in raw.items()}})


def _as_dict(value, question: dict) -> dict:
    if isinstance(value, dict):
        return value
    qtype = question["type"]
    if qtype == "noul":
        return {"type": "noul", "noul": getattr(value, "noul", 0.0)}
    if qtype == "choice":
        return {
            "type": "choice",
            "choice": getattr(value, "choice", ""),
            "probabilities": getattr(value, "probabilities", {}) or {},
            "confidence": getattr(value, "confidence", None),
        }
    return {
        "type": "score",
        "score": getattr(value, "score", 0.0),
        "probabilities": getattr(value, "probabilities", {}) or {},
        "confidence": getattr(value, "confidence", None),
        "legend": getattr(value, "legend", None),
    }


def _jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return repr(value)