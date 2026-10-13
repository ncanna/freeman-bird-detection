"""Cloudflare Clef and Clef-Flash. Release 2026-10-01.

Hosted on Workers AI, Apache-2.0 weights on the Hub. Jev-API compatible, and
unlike Jev it has a vision encoder.

Docs: https://developers.cloudflare.com/workers-ai/models/clef/
Input schema: ``model`` (clef | clef-flash), ``state``, ``questions``, and an
optional ``images`` array of up to 4 embedded PNG/JPEG/WebP images placed
before the state. Remote URLs are not accepted and there is no video field.

    https://api.cloudflare.com/client/v4/accounts/$CLOUDFLARE_ACCOUNT_ID/ai/run/@cf/cloudflare/clef

Needs CLOUDFLARE_AUTH_TOKEN and CLOUDFLARE_ACCOUNT_ID.
"""

from __future__ import annotations

import logging

from system_one_detector.models.base import DecisionBackend
from system_one_detector.models.http import api_key_from_env, post_json, unwrap_result
from system_one_detector.models.media import clef_images
from system_one_detector.models.registry import register_backend
from system_one_detector.util.schema import DecisionResult, parse_systemone_response, validate_questions

logger = logging.getLogger(__name__)

CLEF_URL = (
    "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/cloudflare/clef"
)
CLEF_FLASH_URL = (
    "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/cloudflare/clef-flash"
)

_MAX_IMAGES = 4


@register_backend("clef")
class ClefBackend(DecisionBackend):
    name = "clef"
    sees_pixels = True

    def __init__(
        self,
        *,
        model: str = "clef",
        account_id_env: str = "CLOUDFLARE_ACCOUNT_ID",
        auth_token_env: str = "CLOUDFLARE_AUTH_TOKEN",
        account_id: str | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        if model not in {"clef", "clef-flash"}:
            raise ValueError(f"model must be clef or clef-flash; got {model!r}")
        self.model = model
        self.account_id_env = account_id_env
        self.auth_token_env = auth_token_env
        self.account_id = account_id
        self.timeout_s = timeout_s

    @classmethod
    def from_config(cls, config) -> "ClefBackend":
        return cls(
            model=config.model or "clef",
            account_id_env=config.account_id_env or "CLOUDFLARE_ACCOUNT_ID",
            auth_token_env=config.auth_token_env or "CLOUDFLARE_AUTH_TOKEN",
            account_id=config.account_id,
            timeout_s=config.timeout_s,
        )

    def decide(self, state, questions, *, media_path=None, modality="text") -> DecisionResult:
        questions = validate_questions(questions)
        # Check the modality before credentials, so a schema problem is not
        # hidden behind a missing key.
        if modality == "video":
            raise ValueError(
                "Clef's input schema has images only. No video field is published, so a video "
                "frame cannot be sent as video. Set modality: image and point images_dir at frames."
            )
        if modality == "audio":
            raise ValueError("Clef's input schema has images only; audio is not accepted.")
        if modality != "text" and not media_path:
            raise ValueError(f"Clef modality {modality!r} requires media_path")

        account_id = self.account_id or api_key_from_env(self.account_id_env)
        if not account_id:
            raise RuntimeError(
                f"Set {self.account_id_env} to your Cloudflare account id. "
                "The Workers AI URL needs it."
            )
        token = api_key_from_env(self.auth_token_env)
        if not token:
            raise RuntimeError(f"Set {self.auth_token_env} to a Cloudflare API token with Workers AI access.")

        payload: dict = {"model": self.model, "state": state, "questions": questions}
        if modality != "text":
            payload["images"] = clef_images([media_path])

        template = CLEF_FLASH_URL if self.model == "clef-flash" else CLEF_URL
        url = template.format(account_id=account_id)
        raw = post_json(url, payload, api_key=token, timeout_s=self.timeout_s)
        if raw.get("success") is False:
            errors = raw.get("errors") or []
            detail = "; ".join(str(err.get("message", err)) for err in errors if isinstance(err, dict))
            raise RuntimeError(f"Cloudflare returned success=false: {detail or json_short(raw)}")
        result = parse_systemone_response(unwrap_result(raw))
        if not result.model:
            result.model = self.model
        if raw.get("success") is not None:
            result.usage.setdefault("cloudflare_success", raw["success"])
        return result


def json_short(raw: dict, limit: int = 300) -> str:
    import json

    return json.dumps(raw)[:limit]