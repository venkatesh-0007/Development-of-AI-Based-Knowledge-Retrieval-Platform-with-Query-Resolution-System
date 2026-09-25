"""Centralized Confidence Calculator for Milestone 4.

Eliminates duplicated and divergent confidence heuristics across:
- Retrieval Agent
- Response Generation Agent
- Transparency Panel
- Analytics Storage & Telemetry Tracker
- Knowledge Gap Detection

Provides:
- Mathematical formulation: Combined = weight_top * TopScore + weight_avg * AvgTopK
- Classification into HIGH, MEDIUM, LOW, NONE / INSUFFICIENT
- Configurable threshold definitions
- Robust boundary condition handling
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class ConfidenceLevel(str, Enum):
    """Categorical classification of response groundedness and evidence quality."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NONE = "NONE"


@dataclass
class ConfidenceThresholds:
    """Configurable threshold and weighting parameters."""
    high: float = 0.70
    medium: float = 0.45
    low: float = 0.20
    weight_top: float = 0.70
    weight_avg: float = 0.30

    def __post_init__(self):
        if not (self.high > self.medium > self.low >= 0.0):
            raise ValueError(
                f"Invalid threshold ordering: must satisfy high ({self.high}) > "
                f"medium ({self.medium}) > low ({self.low}) >= 0.0"
            )
        weight_sum = self.weight_top + self.weight_avg
        if abs(weight_sum - 1.0) > 1e-4:
            raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")


@dataclass
class ConfidenceResult:
    """Detailed output from ConfidenceCalculator."""
    combined_score: float
    confidence_level: ConfidenceLevel
    top_chunk_score: float
    avg_top_k_score: float
    is_sufficient: bool

    def to_dict(self):
        return {
            "combined_score": self.combined_score,
            "confidence_level": self.confidence_level.value,
            "top_chunk_score": self.top_chunk_score,
            "avg_top_k_score": self.avg_top_k_score,
            "is_sufficient": self.is_sufficient
        }


class ConfidenceCalculator:
    """Single central authority for confidence score calculation and categorization."""

    def __init__(
        self,
        thresholds: Optional[ConfidenceThresholds] = None,
        high: Optional[float] = None,
        medium: Optional[float] = None,
        low: Optional[float] = None,
        weight_top: Optional[float] = None,
        weight_avg: Optional[float] = None
    ):
        if thresholds is not None:
            self.thresholds = thresholds
        else:
            h = high if high is not None else 0.70
            m = medium if medium is not None else 0.45
            l = low if low is not None else 0.20
            wt = weight_top if weight_top is not None else 0.70
            wa = weight_avg if weight_avg is not None else 0.30
            self.thresholds = ConfidenceThresholds(
                high=h,
                medium=m,
                low=l,
                weight_top=wt,
                weight_avg=wa
            )

    def compute_score(self, top_chunk_score: float, avg_top_k_score: float) -> float:
        """Compute weighted scalar confidence score."""
        top = max(0.0, float(top_chunk_score or 0.0))
        avg = max(0.0, float(avg_top_k_score or 0.0))
        raw = self.thresholds.weight_top * top + self.thresholds.weight_avg * avg
        clamped = min(1.0, max(0.0, raw))
        return round(clamped, 4)

    def classify(self, score: float) -> ConfidenceLevel:
        """Classify a scalar score into HIGH, MEDIUM, LOW, or NONE."""
        s = float(score or 0.0)
        if s >= self.thresholds.high:
            return ConfidenceLevel.HIGH
        elif s >= self.thresholds.medium:
            return ConfidenceLevel.MEDIUM
        elif s >= self.thresholds.low:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.NONE

    def calculate(
        self,
        top_chunk_score: float,
        avg_top_k_score: float,
        min_sufficiency_threshold: Optional[float] = None
    ) -> ConfidenceResult:
        """Execute complete calculation and return ConfidenceResult."""
        combined = self.compute_score(top_chunk_score, avg_top_k_score)
        level = self.classify(combined)

        cutoff = min_sufficiency_threshold if min_sufficiency_threshold is not None else self.thresholds.medium
        is_sufficient = (combined >= cutoff) and (level in (ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM))

        return ConfidenceResult(
            combined_score=combined,
            confidence_level=level,
            top_chunk_score=round(top_chunk_score, 4) if top_chunk_score else 0.0,
            avg_top_k_score=round(avg_top_k_score, 4) if avg_top_k_score else 0.0,
            is_sufficient=is_sufficient
        )


# Global singleton instance with standard calibrated configuration
default_confidence_calculator = ConfidenceCalculator()
