"""ModelAdapterRegistry and @register_adapter decorator."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from experiments.adapters.base import BaseModelAdapter

_REGISTRY: dict[str, type["BaseModelAdapter"]] = {}


def register_adapter(name: str):
    """Class decorator that registers an adapter under the given name."""
    def decorator(cls: type["BaseModelAdapter"]) -> type["BaseModelAdapter"]:
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_adapter(name: str) -> type["BaseModelAdapter"]:
    """Return the adapter class registered under name, or raise KeyError."""
    if name not in _REGISTRY:
        raise KeyError(
            f"No adapter registered for model_name={name!r}. "
            f"Available: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]


def list_adapters() -> list[str]:
    """Return sorted list of registered adapter names."""
    return sorted(_REGISTRY.keys())
