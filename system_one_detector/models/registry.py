"""Backend registry, same idea as ``hlwdetector.registry``."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from system_one_detector.models.base import DecisionBackend
    from system_one_detector.util.config import SystemOneConfig

_REGISTRY: dict[str, type["DecisionBackend"]] = {}


def register_backend(name: str):
    def decorator(cls: type["DecisionBackend"]) -> type["DecisionBackend"]:
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_backend(name: str) -> type["DecisionBackend"]:
    if name not in _REGISTRY:
        raise KeyError(
            f"No System One backend registered for {name!r}. "
            f"Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def list_backends() -> list[str]:
    return sorted(_REGISTRY)


def build_backend(config: "SystemOneConfig") -> "DecisionBackend":
    cls = get_backend(config.backend)
    return cls.from_config(config)
