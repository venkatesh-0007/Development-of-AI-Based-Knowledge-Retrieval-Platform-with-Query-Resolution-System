"""Confidence calculation module for Milestone 4.

Centralizes all confidence score computation and classification across agents,
retrieval, response generation, transparency panels, and analytics.
"""
from .calculator import (
    ConfidenceLevel,
    ConfidenceThresholds,
    ConfidenceCalculator,
    ConfidenceResult,
    default_confidence_calculator
)

__all__ = [
    "ConfidenceLevel",
    "ConfidenceThresholds",
    "ConfidenceCalculator",
    "ConfidenceResult",
    "default_confidence_calculator"
]
