"""OpenAI Decisions API. Announced at DevDay 2026-09-29, limited preview.

There is no published request or response schema. As of 2026-10-02 a standard
API key gets HTTP 403 "Decision API is not enabled for this user" from
``POST /v1/decisions``, the docs pages 404, and no model id contains
"decision". Public reporting says it accepts text or images, returns one of
your predefined answers, and runs on a version of GPT-6 Luna. Reporting also
mentions confidence, which no OpenAI page confirms.

Because the contract is unpublished, this backend sends the System One body
shape that the rest of the ecosystem converges on and reads whichever of the
plausible answer envelopes comes back. Expect to edit ``_build_payload`` and
``_read_answers`` once OpenAI ships the real schema. If you want the same job
today on an OpenAI key, use Luna with a strict enum schema instead.
"""

from __future__ import annotations

import logging

from system_one_detector.models.base import DecisionBackend
from system_one_detector.models.http import api_key_from_env, post_json, unwrap_result
from system_one_detector.models.registry import register_backend
from system_one_detector.util.schema import (
    Answer,
    DecisionResult,
    align_option_probabilities,
    parse_answer,
    validate_questions,
)

logger = logging.getLogger(__name__)

OPENAI_URL = "https://api.openai.com/v1/decisions"


@register_backend("openai_decisions")
class OpenAiDecisionsBackend(DecisionBackend):
    name = "openai_decisions"
    sees_pixels = True

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        endpoint: str = OPENAI_URL,
        timeout_s: float = 120.0,
    ) -> None:
        self.model = model
        self.api_key_env = api_key_env
        self.endpoint = endpoint
        self.timeout_s = timeout_s

    @classmethod
    def from_config(cls, config) -> "OpenAiDecisionsBackend":
        return cls(
            model=config.model,
            api_key_env=config.api_key_env or "OPENAI_API_KEY",
            endpoint=config.endpoint or OPENAI_URL,
            timeout_s=config.timeout_s,
        )

    def _build_payload(self, state, questions: dict[str, dict]) -> dict:
        payload: dict = {"context": state, "questions": questions}
        if self.model:
            payload["model"] = self.model
        return payload

    def _read_answers(self, body: dict, questions: dict[str, dict]) -> dict[str, Answer]:
        raw = unwrap_result(body)
        answers = raw.get("answers")
        if isinstance(answers, dict):
            parsed = {}
            for qid, value in answers.items():
                parsed[qid] = value if isinstance(value, Answer) else parse_answer(value)
            return parsed
        # Fall back to a bare selection. Reporting describes "an answer from
        # your list", so a bare choice may come back instead of a full map.
        for key in ("choice", "answer", "selection", "label", "result"):
            selection = raw.get(key)
            if selection is None:
                continue
            parsed = {}
            for qid, question in questions.items():
                if question["type"] != "choice":
                    parsed[qid] = parse_answer({"type": question["type"], key: selection})
                    continue
                options = list(question["criteria"])
                probs = align_option_probabilities({"prediction": selection}, options)
                if "confidence" in raw and isinstance(raw["confidence"], (int, float)):
                    confidence = float(raw["confidence"])
                else:
                    confidence = max(probs.values())
                parsed[qid] = Answer(
                    type="choice",
                    choice=str(selection),
                    probabilities=probs,
                    confidence=confidence,
                )
            return parsed
        raise RuntimeError(f"no answers in Decisions API response: {sorted(raw)[:12]}")

    def decide(self, state, questions, *, media_path=None, modality="text") -> DecisionResult:
        questions = validate_questions(questions)
        if modality != "text" and not media_path:
            raise ValueError(f"modality {modality!r} requires media_path")
        key = api_key_from_env(self.api_key_env)
        if not key:
            raise RuntimeError(f"Set {self.api_key_env} to an OpenAI API key.")
        payload = self._build_payload(state, questions)
        if modality != "text":
            if modality == "video":
                raise ValueError("Pass a video as frames. OpenAI reported image context for this preview.")
            from system_one_detector.models.media import data_uri

            payload["context"] = [{"type": "image_url", "image_url": {"url": data_uri(media_path)}}]
        logger.warning(
            "OpenAI Decisions API is an undocumented limited preview. The body shape here is a "
            "best guess; adjust _build_payload and _read_answers when the real schema ships."
        )
        raw = post_json(self.endpoint, payload, api_key=key, timeout_s=self.timeout_s)
        answers = self._read_answers(raw, questions)
        return DecisionResult(
            model=self.model or "decisions-api",
            answers=answers,
            usage=raw.get("usage") if isinstance(raw.get("usage"), dict) else {},
            raw=raw,
        )