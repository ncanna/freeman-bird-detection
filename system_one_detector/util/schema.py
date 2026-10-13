"""System One question and answer shapes.

The wire format follows the TypeSafe evaluation endpoint
(POST /v1/systemone): a state plus a map of noul, choice, and score
questions. Laya returns the same answer keys. Jev-Omni does not; its
adapter translates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


QUESTION_TYPES = ("noul", "choice", "score")


@dataclass
class Answer:
    """One typed answer, normalized across backends."""

    type: str
    noul: float | None = None
    choice: str | None = None
    score: float | None = None
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float | None = None
    legend: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return {key: value for key, value in data.items() if value is not None and value != {}}


@dataclass
class DecisionResult:
    model: str
    answers: dict[str, Answer]
    usage: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "answers": {qid: answer.to_dict() for qid, answer in self.answers.items()},
            "usage": self.usage,
        }


def instructions_text(instructions: Any) -> str:
    """Flatten structured instructions to a single string for backends that need one."""
    if isinstance(instructions, str):
        return instructions
    if instructions is None:
        return ""
    import json

    return json.dumps(instructions, ensure_ascii=False)


def validate_questions(questions: dict[str, dict]) -> dict[str, dict]:
    """Return a copy of questions, or raise ValueError naming the bad field."""
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a non-empty map of id -> question")
    cleaned: dict[str, dict] = {}
    for qid, question in questions.items():
        if not isinstance(question, dict):
            raise ValueError(f"question {qid!r} must be an object")
        qtype = question.get("type")
        if qtype not in QUESTION_TYPES:
            raise ValueError(f"question {qid!r} has type {qtype!r}; expected one of {QUESTION_TYPES}")
        if not question.get("instructions"):
            raise ValueError(f"question {qid!r} is missing instructions")
        if qtype == "choice":
            criteria = question.get("criteria")
            if not isinstance(criteria, dict) or len(criteria) < 2:
                raise ValueError(f"choice question {qid!r} needs criteria with at least 2 options")
            if len(criteria) > 255:
                raise ValueError(f"choice question {qid!r} has {len(criteria)} options; the API max is 255")
        if qtype == "score":
            criteria = question.get("criteria")
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                raise ValueError(f"score question {qid!r} needs 2 to 10 ordered criteria")
        cleaned[qid] = question
    return cleaned


def _as_float(value: Any) -> float:
    return float(value)


def _prob_map(raw: Any) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    return {str(key): _as_float(value) for key, value in raw.items()}


def parse_answer(raw: dict[str, Any]) -> Answer:
    """Parse one TypeSafe/Laya answer object. Missing ``type`` is inferred."""
    if not isinstance(raw, dict):
        raise ValueError(f"answer must be an object, got {type(raw).__name__}")
    qtype = raw.get("type")
    if qtype is None:
        if "noul" in raw:
            qtype = "noul"
        elif "choice" in raw:
            qtype = "choice"
        elif "score" in raw:
            qtype = "score"
        else:
            raise ValueError(f"cannot infer answer type from keys {sorted(raw)}")
    if qtype == "noul":
        return Answer(type="noul", noul=_as_float(raw["noul"]))
    if qtype == "choice":
        return Answer(
            type="choice",
            choice=str(raw["choice"]),
            probabilities=_prob_map(raw.get("probabilities")),
            confidence=_as_float(raw["confidence"]) if raw.get("confidence") is not None else None,
        )
    if qtype == "score":
        legend = raw.get("legend")
        return Answer(
            type="score",
            score=_as_float(raw["score"]),
            probabilities=_prob_map(raw.get("probabilities")),
            confidence=_as_float(raw["confidence"]) if raw.get("confidence") is not None else None,
            legend={str(k): str(v) for k, v in legend.items()} if isinstance(legend, dict) else None,
        )
    raise ValueError(f"unknown answer type {qtype!r}")


def parse_systemone_response(payload: dict[str, Any]) -> DecisionResult:
    """Parse a POST /v1/systemone JSON body."""
    answers_raw = payload.get("answers")
    if not isinstance(answers_raw, dict):
        raise ValueError("systemone response is missing an answers object")
    answers = {qid: parse_answer(raw) for qid, raw in answers_raw.items()}
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    model = str(payload.get("model") or "")
    return DecisionResult(model=model, answers=answers, usage=usage, raw=payload)


def align_option_probabilities(result: dict[str, Any], options: list[str]) -> dict[str, float]:
    """Map a Jev-Omni-style result onto the option strings we sent.

    Accepts a dict keyed by option or by index, a list aligned with
    ``options``, or a one-hot fallback on ``prediction``.
    """
    probs = result.get("probabilities")
    if isinstance(probs, dict):
        aligned: dict[str, float] = {}
        for index, option in enumerate(options):
            if option in probs:
                aligned[option] = _as_float(probs[option])
            elif str(index) in probs:
                aligned[option] = _as_float(probs[str(index)])
            else:
                aligned[option] = 0.0
        total = sum(aligned.values())
        if total > 0:
            return aligned
    if isinstance(probs, list) and len(probs) >= len(options):
        return {option: _as_float(prob) for option, prob in zip(options, probs)}
    prediction = result.get("prediction")
    if prediction in options:
        return {option: 1.0 if option == prediction else 0.0 for option in options}
    index = result.get("prediction_index")
    if isinstance(index, int) and 0 <= index < len(options):
        return {option: 1.0 if i == index else 0.0 for i, option in enumerate(options)}
    raise ValueError(f"could not align probabilities onto options {options}")
