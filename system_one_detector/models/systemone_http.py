"""Generic POST /v1/systemone client.

Use this for servers that speak a System One request shape and are not the
official Jev API: ``laya-serve``, Kev, Decider, Clef, Ollama, or OneJev.

OneJev documents an extra ``media`` field. Set ``media_style: onejev`` to send
a data-URI image plus an ``<image:1>`` placeholder. Ollama's local
``/v1/systemone`` endpoint takes an ``images`` array of raw base64 strings;
select ``media_style: ollama``. Both formats send the full question map in one
HTTP request. Text-only servers should leave ``send_media`` false.
"""

from __future__ import annotations

import logging

from system_one_detector.models.base import DecisionBackend
from system_one_detector.models.http import api_key_from_env, post_json, resolve_endpoint
from system_one_detector.models.media import base64_image, onejev_media, state_with_image_placeholder
from system_one_detector.models.registry import register_backend
from system_one_detector.util.schema import DecisionResult, parse_systemone_response, validate_questions

logger = logging.getLogger(__name__)


@register_backend("systemone_http")
class SystemOneHttpBackend(DecisionBackend):
    name = "systemone_http"
    sees_pixels = False

    def __init__(
        self,
        *,
        base_url: str | None = None,
        endpoint: str | None = None,
        model: str | None = None,
        api_key_env: str = "TYPESAFE_API_KEY",
        timeout_s: float = 120.0,
        send_media: bool = False,
        media_style: str = "onejev",
    ) -> None:
        self.endpoint = resolve_endpoint(base_url, endpoint)
        self.model = model
        self.api_key_env = api_key_env
        self.timeout_s = timeout_s
        self.send_media = send_media
        self.media_style = media_style
        self.sees_pixels = bool(send_media)

    @classmethod
    def from_config(cls, config) -> "SystemOneHttpBackend":
        return cls(
            base_url=config.base_url,
            endpoint=config.endpoint,
            model=config.model,
            api_key_env=config.api_key_env,
            timeout_s=config.timeout_s,
            send_media=config.send_media,
            media_style=config.media_style,
        )

    def decide(self, state, questions, *, media_path=None, modality="text") -> DecisionResult:
        questions = validate_questions(questions)
        payload_state = state
        payload: dict = {"state": payload_state, "questions": questions}
        if self.model:
            payload["model"] = self.model
        if self.send_media:
            if not media_path:
                raise ValueError("send_media is set but this frame has no media_path")
            if self.media_style == "ollama":
                if modality != "image":
                    raise ValueError("Ollama System One image support requires modality='image'")
                payload["images"] = [base64_image(media_path)]
            elif self.media_style == "onejev":
                payload_state = state_with_image_placeholder(state)
                payload["state"] = payload_state
                payload["media"] = onejev_media(media_path, modality)
            else:
                raise ValueError(
                    f"unsupported media_style {self.media_style!r}. "
                    "Use 'onejev' or 'ollama'."
                )
        elif media_path:
            logger.info("systemone_http is not sending media. Set send_media: true for OneJev-style servers.")
        key = api_key_from_env(self.api_key_env)
        raw = post_json(self.endpoint, payload, api_key=key, timeout_s=self.timeout_s)
        result = parse_systemone_response(raw)
        if not result.model:
            result.model = self.model or self.endpoint
        return result
