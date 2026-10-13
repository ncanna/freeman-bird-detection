"""TypeSafe Jev. Hosted, text only.

Docs: https://docs.typesafe.ai/api
Endpoint: POST https://api.typesafe.ai/v1/systemone
Jev does not accept images, audio, or video. A frame path is not sent.
"""

from __future__ import annotations

import logging

from system_one_detector.models.base import DecisionBackend
from system_one_detector.models.http import api_key_from_env, post_json
from system_one_detector.models.registry import register_backend
from system_one_detector.util.schema import DecisionResult, parse_systemone_response, validate_questions

logger = logging.getLogger(__name__)

JEV_URL = "https://api.typesafe.ai/v1/systemone"


@register_backend("jev")
class JevBackend(DecisionBackend):
    name = "jev"
    sees_pixels = False

    def __init__(
        self,
        *,
        model: str = "jev-latest",
        api_key_env: str = "TYPESAFE_API_KEY",
        timeout_s: float = 120.0,
        endpoint: str = JEV_URL,
    ) -> None:
        self.model = model
        self.api_key_env = api_key_env
        self.timeout_s = timeout_s
        self.endpoint = endpoint
        self._warned = False

    @classmethod
    def from_config(cls, config) -> "JevBackend":
        return cls(
            model=config.model or "jev-latest",
            api_key_env=config.api_key_env or "TYPESAFE_API_KEY",
            timeout_s=config.timeout_s,
            endpoint=config.endpoint or JEV_URL,
        )

    def decide(self, state, questions, *, media_path=None, modality="text") -> DecisionResult:
        if media_path and not self._warned:
            logger.warning(
                "Jev is text-only (TypeSafe docs: images are not supported). "
                "The frame file is not sent; only the text state is."
            )
            self._warned = True
        key = api_key_from_env(self.api_key_env)
        if not key:
            raise RuntimeError(
                f"Set {self.api_key_env} to a TypeSafe API key. "
                "Keys are created at https://console.typesafe.ai/keys"
            )
        payload = {
            "state": state,
            "model": self.model,
            "questions": validate_questions(questions),
        }
        raw = post_json(self.endpoint, payload, api_key=key, timeout_s=self.timeout_s)
        result = parse_systemone_response(raw)
        if not result.model:
            result.model = self.model
        return result
