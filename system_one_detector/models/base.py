"""Backend interface shared by Jev, Laya, Jev-Omni, and HTTP servers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from system_one_detector.util.schema import DecisionResult


class DecisionBackend(ABC):
    """Ask typed questions about one state.

    ``sees_pixels`` is False for text-only models. The engine still runs them,
    but it will not pretend a file path is a picture.
    """

    name: str
    sees_pixels: bool = False

    @classmethod
    def from_config(cls, config: Any) -> "DecisionBackend":
        return cls()

    @abstractmethod
    def decide(
        self,
        state: Any,
        questions: dict[str, dict],
        *,
        media_path: str | None = None,
        modality: str = "text",
    ) -> DecisionResult:
        """Return one normalized result. Implementations may call out or load weights."""
