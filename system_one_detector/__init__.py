"""System One decision experiments on Freeman camera-trap frames.

These models (TypeSafe Jev, Laya, Jev-Omni, and other ``/v1/systemone``
servers) return calibrated probabilities for typed questions. They do not
regress bounding boxes. This package turns detection labels into those
questions and scores the answers.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
