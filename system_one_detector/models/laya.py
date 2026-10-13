"""Laya, a local non-autoregressive System One model.

``laya.Router.predict(state, questions)`` and ``laya.load(...).predict``
return the same answer keys as TypeSafe. Laya reads text, not pixels.
Install with ``pip install laya``. Checkpoint ids:

- convaiinnovations/laya                  English, ModernBERT-large
- convaiinnovations/laya-multilingual     100+ languages
- convaiinnovations/laya-typed-decisions  specialist; not a silent default
"""

from __future__ import annotations

import logging

from system_one_detector.models.base import DecisionBackend
from system_one_detector.models.registry import register_backend
from system_one_detector.util.schema import DecisionResult, parse_systemone_response, validate_questions

logger = logging.getLogger(__name__)


@register_backend("laya")
class LayaBackend(DecisionBackend):
    name = "laya"
    sees_pixels = False

    def __init__(
        self,
        *,
        checkpoint: str = "convaiinnovations/laya",
        subfolder: str | None = None,
        use_router: bool = True,
        model: str | None = None,
        device: str | None = None,
    ) -> None:
        self.checkpoint = checkpoint
        self.subfolder = subfolder
        self.use_router = use_router
        self.model = model
        self.device = device
        self._runner = None
        self._warned = False

    @classmethod
    def from_config(cls, config) -> "LayaBackend":
        return cls(
            checkpoint=config.checkpoint or "convaiinnovations/laya",
            subfolder=config.subfolder,
            use_router=config.use_router,
            model=config.model,
            device=config.device,
        )

    def _load(self):
        if self._runner is not None:
            return self._runner
        try:
            import laya
        except ImportError as exc:
            raise RuntimeError("Laya is not installed. Run `pip install laya`.") from exc
        if self.use_router:
            from laya import Router

            kwargs = {}
            if self.device:
                kwargs["device"] = self.device
            self._runner = Router(**kwargs)
            return self._runner
        load_kwargs = {}
        if self.subfolder:
            load_kwargs["subfolder"] = self.subfolder
        self._runner = laya.load(self.checkpoint, **load_kwargs)
        return self._runner

    def decide(self, state, questions, *, media_path=None, modality="text") -> DecisionResult:
        if media_path and not self._warned:
            logger.warning("Laya reads the text state only. The frame file is not passed in.")
            self._warned = True
        questions = validate_questions(questions)
        runner = self._load()
        if self.use_router and self.model:
            raw = runner.predict(state, questions, model=self.model)
        else:
            raw = runner.predict(state, questions)
        if not isinstance(raw, dict) or "answers" not in raw:
            raise RuntimeError(f"unexpected Laya result keys: {sorted(raw) if isinstance(raw, dict) else type(raw)}")
        result = parse_systemone_response(raw)
        if not result.model:
            routing = raw.get("routing") if isinstance(raw.get("routing"), dict) else {}
            result.model = str(routing.get("repo") or routing.get("model") or self.checkpoint)
        result.raw = raw
        return result
