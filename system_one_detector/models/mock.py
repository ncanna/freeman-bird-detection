"""Deterministic backend for dry runs. Does not load a model or call an API."""

from __future__ import annotations

from system_one_detector.models.base import DecisionBackend
from system_one_detector.models.registry import register_backend
from system_one_detector.util.schema import Answer, DecisionResult, validate_questions


@register_backend("mock")
class MockBackend(DecisionBackend):
    name = "mock"
    sees_pixels = False

    def decide(self, state, questions, *, media_path=None, modality="text") -> DecisionResult:
        questions = validate_questions(questions)
        answers: dict[str, Answer] = {}
        for qid, question in questions.items():
            qtype = question["type"]
            if qtype == "noul":
                answers[qid] = Answer(type="noul", noul=0.5)
            elif qtype == "choice":
                keys = list(question["criteria"])
                uniform = 1.0 / len(keys)
                answers[qid] = Answer(
                    type="choice",
                    choice=keys[0],
                    probabilities={key: uniform for key in keys},
                    confidence=uniform,
                )
            else:
                levels = list(question["criteria"])
                uniform = 1.0 / len(levels)
                answers[qid] = Answer(
                    type="score",
                    score=(len(levels) - 1) / 2.0,
                    probabilities={str(i): uniform for i in range(len(levels))},
                    confidence=uniform,
                    legend={str(i): str(level) for i, level in enumerate(levels)},
                )
        return DecisionResult(model="mock", answers=answers, usage={"media_seen": bool(media_path)})
