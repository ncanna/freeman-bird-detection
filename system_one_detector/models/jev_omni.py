"""Jev-Omni: a multimodal decision classifier, not TypeSafe Jev.

Weights: https://huggingface.co/akhilaaa3/Jev-Omni
Independent of TypeSafe. One question per ``predict`` call. The published
loader requires a CUDA GPU and accepts text, image, audio, and video.

``predict`` returns prediction, prediction_index, confidence, probabilities
for a flat option list. This module translates noul / choice / score into
that call and back into the System One answer shape.
"""

from __future__ import annotations

import logging
from typing import Any

from system_one_detector.models.base import DecisionBackend
from system_one_detector.models.registry import register_backend
from system_one_detector.util.schema import (
    Answer,
    DecisionResult,
    align_option_probabilities,
    instructions_text,
    validate_questions,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "akhilaaa3/Jev-Omni"


def omni_call(question: dict) -> tuple[str, list[str]]:
    """Return ``(question_text, options)`` for one System One question.

    Choice options are the criteria keys. Descriptions are appended to the
    question because Jev-Omni has a flat option list, not key/description pairs.
    """
    qtype = question["type"]
    text = instructions_text(question.get("instructions"))
    if qtype == "noul":
        criteria = question.get("criteria") or {}
        yes = criteria.get("true")
        no = criteria.get("false")
        if yes or no:
            text = f"{text}\nYes means: {yes or 'the statement is true'}\nNo means: {no or 'the statement is false'}"
        return text, ["No", "Yes"]
    if qtype == "choice":
        criteria = question["criteria"]
        lines = [text, "Options:"]
        for key, description in criteria.items():
            if description:
                lines.append(f"- {key}: {description}")
            else:
                lines.append(f"- {key}")
        return "\n".join(lines), list(criteria.keys())
    if qtype == "score":
        criteria = list(question["criteria"])
        lines = [text, "Levels, from low to high:"]
        for index, level in enumerate(criteria):
            lines.append(f"- {index}: {level}")
        return "\n".join(lines), [str(level) for level in criteria]
    raise ValueError(f"unsupported question type {qtype!r}")


def omni_to_answer(question: dict, result: dict[str, Any]) -> Answer:
    """Convert one Jev-Omni result into a System One answer."""
    qtype = question["type"]
    _text, options = omni_call(question)
    probs = align_option_probabilities(result, options)
    confidence = result.get("confidence")
    confidence_f = float(confidence) if confidence is not None else None
    if qtype == "noul":
        yes = float(probs.get("Yes", 0.0))
        return Answer(
            type="noul",
            noul=yes,
            probabilities={"true": yes, "false": float(probs.get("No", 0.0))},
            confidence=confidence_f,
        )
    if qtype == "choice":
        choice = result.get("prediction")
        if choice not in options:
            choice = max(probs, key=probs.get)
        return Answer(type="choice", choice=str(choice), probabilities=probs, confidence=confidence_f)
    levels = [str(level) for level in question["criteria"]]
    indexed = {str(i): float(probs.get(level, 0.0)) for i, level in enumerate(levels)}
    score = sum(i * indexed[str(i)] for i in range(len(levels)))
    return Answer(
        type="score",
        score=score,
        probabilities=indexed,
        confidence=confidence_f,
        legend={str(i): levels[i] for i in range(len(levels))},
    )


@register_backend("jev_omni")
class JevOmniBackend(DecisionBackend):
    name = "jev_omni"
    sees_pixels = True

    def __init__(self, *, model_id: str = DEFAULT_MODEL_ID, device: str = "cuda") -> None:
        self.model_id = model_id
        self.device = device
        self._classifier = None

    @classmethod
    def from_config(cls, config) -> "JevOmniBackend":
        return cls(model_id=config.model or DEFAULT_MODEL_ID, device=config.device or "cuda")

    def _load(self):
        if self._classifier is not None:
            return self._classifier
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "huggingface_hub is required for Jev-Omni. "
                "Install the requirements linked from akhilaaa3/Jev-Omni."
            ) from exc
        import sys

        path = snapshot_download(self.model_id)
        if path not in sys.path:
            sys.path.insert(0, path)
        from jev_omni import load_jev_omni

        logger.info("Loading Jev-Omni (%s) on %s. The upstream loader requires CUDA.", self.model_id, self.device)
        self._classifier = load_jev_omni(model_id=self.model_id, device=self.device)
        return self._classifier

    def decide(self, state, questions, *, media_path=None, modality="text") -> DecisionResult:
        questions = validate_questions(questions)
        if modality != "text" and not media_path:
            raise ValueError(f"Jev-Omni modality {modality!r} requires media_path")
        classifier = self._load()
        state_text = state if isinstance(state, str) else _state_text(state)
        answers = {}
        raw_calls = []
        for qid, question in questions.items():
            question_text, options = omni_call(question)
            kwargs = {
                "state": state_text,
                "question": question_text,
                "options": options,
            }
            if modality != "text":
                kwargs["media"] = media_path
                kwargs["modality"] = modality
            result = classifier.predict(**kwargs)
            if not isinstance(result, dict):
                raise RuntimeError(f"Jev-Omni returned {type(result).__name__}, expected a dict")
            answers[qid] = omni_to_answer(question, result)
            raw_calls.append({"question_id": qid, "options": options, "result": result})
        return DecisionResult(model=self.model_id, answers=answers, raw={"calls": raw_calls})


def _state_text(state) -> str:
    import json

    return json.dumps(state, ensure_ascii=False)
