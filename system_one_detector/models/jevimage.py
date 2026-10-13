"""jevimage. One dual-encoder pass per image, then every question is a matmul.

Not a VLM. A frozen CLIP/SigLIP-style image tower and a text tower produce a
single unit-norm vector; choice, score, noul, and trained ``head`` questions
are scored against it. The tenth question about a frame costs a matmul rather
than another forward pass once its captions are cached.

Install:
    pip install jevimage                 # HTTP client only
    pip install 'jevimage[local]'        # encoder in this process (torch, transformers)
    pip install 'jevimage[serve]'        # run a local server for others

Set ``base_url`` to ``connect()`` to a server, or leave it empty to ``load()``
locally. Also has a ``head`` question type trained on your own labelled images,
which this harness does not build labels for; see ``train_head`` below.
"""

from __future__ import annotations

import logging

from system_one_detector.models.base import DecisionBackend
from system_one_detector.models.registry import register_backend
from system_one_detector.util.schema import Answer, DecisionResult, validate_questions

logger = logging.getLogger(__name__)


def _field(raw, name, default=None):
    """Read a field from a jevimage Answer (dict subclass) or a plain dict."""
    if isinstance(raw, dict):
        return raw.get(name, default)
    return getattr(raw, name, default)


def _to_answer(raw, question: dict) -> Answer:
    qtype = question["type"]
    if qtype == "noul":
        return Answer(
            type="noul",
            noul=float(_field(raw, "noul", 0.0) or 0.0),
            confidence=_field(raw, "confidence"),
        )
    if qtype == "choice":
        probabilities = _field(raw, "probabilities") or {}
        return Answer(
            type="choice",
            choice=str(_field(raw, "choice")),
            probabilities={str(k): float(v) for k, v in probabilities.items()},
            confidence=_field(raw, "confidence"),
        )
    if qtype == "score":
        probabilities = _field(raw, "probabilities") or {}
        legend = _field(raw, "legend") or {}
        return Answer(
            type="score",
            score=float(_field(raw, "score", 0.0) or 0.0),
            probabilities={str(k): float(v) for k, v in probabilities.items()},
            confidence=_field(raw, "confidence"),
            legend={str(k): str(v) for k, v in legend.items()} if legend else None,
        )
    raise ValueError(f"jevimage backend does not handle question type {qtype!r}")


@register_backend("jevimage")
class JevImageBackend(DecisionBackend):
    name = "jevimage"
    sees_pixels = True

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key_env: str = "JEVIMAGE_API_KEY",
        encoder: str | None = None,
    ) -> None:
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.encoder = encoder
        self._agent = None

    @classmethod
    def from_config(cls, config) -> "JevImageBackend":
        return cls(base_url=config.base_url, encoder=config.encoder)

    def _load(self):
        if self._agent is not None:
            return self._agent
        try:
            import jevimage
        except ImportError as exc:
            raise RuntimeError(
                "jevimage is not installed. Use `pip install jevimage` for a server, "
                "or `pip install 'jevimage[local]'` to run the encoder in this process."
            ) from exc
        import os

        if self.base_url:
            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                logger.info("No %s set; connecting to %s without a key.", self.api_key_env, self.base_url)
            self._agent = jevimage.connect(self.base_url, api_key=api_key)
        else:
            self._agent = jevimage.load()
        if self.encoder:
            logger.info("jevimage encoder override requested: %s", self.encoder)
        return self._agent

    def decide(self, state, questions, *, media_path=None, modality="text") -> DecisionResult:
        questions = validate_questions(questions)
        if not media_path:
            raise ValueError("jevimage reads the image file itself. media_path is required.")
        if modality not in {"image", "text"}:
            raise ValueError(f"jevimage is image-only; got modality {modality!r}")
        agent = self._load()
        payload = {qid: dict(question) for qid, question in questions.items()}
        raw = agent.ask(media_path, payload)
        answers = {qid: _to_answer(raw[qid], questions[qid]) for qid in questions}
        return DecisionResult(model=self.encoder or "jevimage", answers=answers, raw={"n_questions": len(answers)})