"""Decision backends. Importing this package registers the built-in ones."""

from system_one_detector.models.base import DecisionBackend, DecisionResult
from system_one_detector.models.registry import build_backend, get_backend, list_backends, register_backend

# Importing each module applies @register_backend.
from system_one_detector.models import clef as _clef  # noqa: F401
from system_one_detector.models import djev as _djev  # noqa: F401
from system_one_detector.models import jev as _jev  # noqa: F401
from system_one_detector.models import jev_omni as _jev_omni  # noqa: F401
from system_one_detector.models import jevimage as _jevimage  # noqa: F401
from system_one_detector.models import laya as _laya  # noqa: F401
from system_one_detector.models import mock as _mock  # noqa: F401
from system_one_detector.models import omni_jev as _omni_jev  # noqa: F401
from system_one_detector.models import openai_decisions as _openai_decisions  # noqa: F401
from system_one_detector.models import systemone_http as _systemone_http  # noqa: F401

__all__ = [
    "DecisionBackend",
    "DecisionResult",
    "build_backend",
    "get_backend",
    "list_backends",
    "register_backend",
]