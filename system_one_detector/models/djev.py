"""Djev. DiffusionGemma reading decisions in one denoise step.

Hosted preview at https://api.djev.dev, invite-only. The open implementation is
mmastrac/djev. See https://api.djev.dev/docs for the contract.

Notes that matter here:

- Endpoint is ``POST /v1/request``, not ``/v1/systemone``.
- It may answer ``200`` directly or ``202`` with a receipt. A 202 receipt is
  not a result; the terminal envelope arrives from ``GET /v1/requests/{id}``.
  ``HTTP 200`` alone does not mean inference succeeded, so read ``status``.
- Images go in a top-level ``images`` array as one data URL, not ``media``.
- Execution is a diffusion sample, so ``seed`` matters. Hold state, image bytes,
  question wording, option order, and options fixed when comparing runs.
- Question instructions and option descriptions may be ``{"image": ..., "text": ...}``.
  The backend leaves that pass-through alone.
"""

from __future__ import annotations

import logging
import time
import uuid

from system_one_detector.models.base import DecisionBackend
from system_one_detector.models.http import api_key_from_env, get_json, post_json_raw, unwrap_result
from system_one_detector.models.media import djev_images
from system_one_detector.models.registry import register_backend
from system_one_detector.util.schema import DecisionResult, parse_systemone_response, validate_questions

logger = logging.getLogger(__name__)

DJEV_URL = "https://api.djev.dev/v1/request"


@register_backend("djev")
class DjevBackend(DecisionBackend):
    name = "djev"
    sees_pixels = True

    def __init__(
        self,
        *,
        model: str = "djev",
        api_key_env: str = "DJEV_API_KEY",
        base_url: str | None = None,
        timeout_s: float = 120.0,
        seed: int | None = 0,
        samples: int = 1,
        isolation: str = "joint",
        score_mode: str = "categorical",
        poll_interval_s: float = 3.0,
        poll_timeout_s: float = 900.0,
    ) -> None:
        self.model = model
        self.api_key_env = api_key_env
        self.base_url = (base_url or "https://api.djev.dev").rstrip("/")
        self.timeout_s = timeout_s
        self.seed = seed
        self.samples = samples
        self.isolation = isolation
        self.score_mode = score_mode
        self.poll_interval_s = poll_interval_s
        self.poll_timeout_s = poll_timeout_s

    @classmethod
    def from_config(cls, config) -> "DjevBackend":
        return cls(
            model=config.model or "djev",
            api_key_env=config.api_key_env or "DJEV_API_KEY",
            base_url=config.base_url,
            timeout_s=config.timeout_s,
            seed=config.seed,
            samples=config.samples,
            isolation=config.isolation,
            score_mode=config.score_mode,
        )

    def _options(self) -> dict:
        options: dict = {"isolation": self.isolation, "score_mode": self.score_mode, "samples": self.samples}
        if self.seed is not None:
            options["seed"] = self.seed
        return options

    def decide(self, state, questions, *, media_path=None, modality="text") -> DecisionResult:
        questions = validate_questions(questions)
        # Input-shape checks first, so a schema problem is not hidden behind a
        # missing key.
        if len(questions) > 32:
            raise ValueError(f"Djev takes 1 to 32 questions per request; got {len(questions)}")
        if modality in {"audio", "video"}:
            raise ValueError(
                f"Djev takes images, not {modality}. Its docs have no video-stream endpoint; "
                "a webcam frame is an ordinary image request."
            )
        if modality != "text" and not media_path:
            raise ValueError(f"Djev modality {modality!r} requires media_path")

        key = api_key_from_env(self.api_key_env)
        if not key:
            raise RuntimeError(
                f"Set {self.api_key_env} to a Djev API key. Access is invite-only; "
                "see https://djev.dev/#workbench"
            )

        payload: dict = {"model": self.model, "state": state, "questions": questions, "options": self._options()}
        if modality != "text":
            payload["images"] = djev_images(media_path)

        operation_id = uuid.uuid4().hex
        body, status, _ = post_json_raw(
            f"{self.base_url}/v1/request",
            payload,
            api_key=key,
            timeout_s=self.timeout_s,
            headers={"Idempotency-Key": operation_id},
            accepted_status=(200, 202),
        )
        if status == 200:
            response = body
        elif status == 202:
            response = self._await_terminal(body, key)
        else:
            raise RuntimeError(f"Djev returned HTTP {status}: {body.get('error', body)}")

        parsed = parse_systemone_response(unwrap_result(response))
        if not parsed.model:
            parsed.model = self.model
        parsed.usage.setdefault("operation_id", operation_id)
        return parsed

    def _await_terminal(self, receipt: dict, key: str) -> dict:
        request_id = receipt.get("id")
        if not request_id:
            raise RuntimeError(f"Djev returned 202 without an id: {receipt}")
        expected_sha = receipt.get("request_sha256")
        deadline = time.monotonic() + self.poll_timeout_s
        url = f"{self.base_url}/v1/requests/{request_id}"
        while time.monotonic() < deadline:
            job, status = get_json(url, api_key=key, timeout_s=self.timeout_s, accepted_status=(200, 202))
            if expected_sha and job.get("request_sha256") not in (None, expected_sha):
                raise RuntimeError("Djev returned a receipt for different request bytes")
            if status == 200:
                job_status = job.get("status")
                if job_status == "succeeded":
                    return job.get("response") or {}
                if job_status == "failed":
                    error = job.get("error") or {}
                    raise RuntimeError(f"Djev job failed: {error.get('message', error)}")
                raise RuntimeError(f"unexpected terminal Djev status {job_status!r}")
            wait = job.get("retry_after", self.poll_interval_s)
            time.sleep(max(self.poll_interval_s, float(wait)))
        raise TimeoutError(
            f"Stopped waiting on Djev job {request_id}. The job is still accepted; "
            "resume it with GET /v1/requests/{id} using the same account."
        )