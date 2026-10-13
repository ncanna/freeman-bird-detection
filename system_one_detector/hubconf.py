"""Builders, in the same role as ``detr_detector/hubconf.py``.

None of these download weights. Jev, Clef, Djev, and OpenAI are API clients.
Laya, Jev-Omni, jevimage, and OmniJev load on the first ``decide`` call.
"""

from __future__ import annotations


def jev(model: str = "jev-latest"):
    from system_one_detector.models.jev import JevBackend

    return JevBackend(model=model)


def laya(checkpoint: str = "convaiinnovations/laya", use_router: bool = True, model: str | None = None):
    from system_one_detector.models.laya import LayaBackend

    return LayaBackend(checkpoint=checkpoint, use_router=use_router, model=model)


def jev_omni(model: str = "akhilaaa3/Jev-Omni", device: str = "cuda"):
    from system_one_detector.models.jev_omni import JevOmniBackend

    return JevOmniBackend(model_id=model, device=device)


def systemone_http(base_url: str, model: str | None = None, send_media: bool = False):
    from system_one_detector.models.systemone_http import SystemOneHttpBackend

    return SystemOneHttpBackend(base_url=base_url, model=model, send_media=send_media)


def clef(model: str = "clef"):
    """Cloudflare Clef or Clef-Flash. Images only. Needs a Cloudflare token."""
    from system_one_detector.models.clef import ClefBackend

    return ClefBackend(model=model)


def djev(model: str = "djev", **options):
    """Djev hosted preview. Images, no video. Invite-only."""
    from system_one_detector.models.djev import DjevBackend

    return DjevBackend(model=model, **options)


def jevimage(base_url: str | None = None, encoder: str | None = None):
    """One encoder pass per image, then matmuls. Not a VLM."""
    from system_one_detector.models.jevimage import JevImageBackend

    return JevImageBackend(base_url=base_url, encoder=encoder)


def omni_jev(adapter_dir: str, base_model_dir: str, noul_bias: float | None = None):
    """tinnel123 OmniJev v1.1: a 4B Qwen3.5 LoRA plus decision heads."""
    from system_one_detector.models.omni_jev import OmniJevBackend

    return OmniJevBackend(adapter_dir=adapter_dir, base_model_dir=base_model_dir, noul_bias=noul_bias)


def openai_decisions(model: str | None = None):
    """OpenAI Decisions API. Undocumented limited preview; may 403."""
    from system_one_detector.models.openai_decisions import OpenAiDecisionsBackend

    return OpenAiDecisionsBackend(model=model)
