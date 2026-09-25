import sys
import unittest
from pathlib import Path

m4_root = str(Path(__file__).resolve().parent.parent)
if m4_root not in sys.path:
    sys.path.insert(0, m4_root)

from app.confidence.calculator import (
    ConfidenceLevel,
    ConfidenceThresholds,
    ConfidenceCalculator,
    default_confidence_calculator
)


class TestConfidenceCalculator(unittest.TestCase):
    """Verifies boundary values, custom thresholds, and edge cases."""

    def setUp(self):
        self.calc = ConfidenceCalculator()

    def test_default_weights_and_thresholds(self):
        self.assertEqual(self.calc.thresholds.high, 0.70)
        self.assertEqual(self.calc.thresholds.medium, 0.45)
        self.assertEqual(self.calc.thresholds.low, 0.20)
        self.assertAlmostEqual(self.calc.thresholds.weight_top, 0.70)
        self.assertAlmostEqual(self.calc.thresholds.weight_avg, 0.30)

    def test_exact_boundary_classifications(self):
        # High boundary
        self.assertEqual(self.calc.classify(0.70), ConfidenceLevel.HIGH)
        self.assertEqual(self.calc.classify(0.85), ConfidenceLevel.HIGH)
        self.assertEqual(self.calc.classify(1.0), ConfidenceLevel.HIGH)

        # Just below high -> Medium
        self.assertEqual(self.calc.classify(0.6999), ConfidenceLevel.MEDIUM)
        self.assertEqual(self.calc.classify(0.45), ConfidenceLevel.MEDIUM)
        self.assertEqual(self.calc.classify(0.55), ConfidenceLevel.MEDIUM)

        # Just below medium -> Low
        self.assertEqual(self.calc.classify(0.4499), ConfidenceLevel.LOW)
        self.assertEqual(self.calc.classify(0.20), ConfidenceLevel.LOW)
        self.assertEqual(self.calc.classify(0.30), ConfidenceLevel.LOW)

        # Below low -> None
        self.assertEqual(self.calc.classify(0.1999), ConfidenceLevel.NONE)
        self.assertEqual(self.calc.classify(0.0), ConfidenceLevel.NONE)
        self.assertEqual(self.calc.classify(-0.1), ConfidenceLevel.NONE)

    def test_compute_score_formula(self):
        # 0.70 * 0.8 + 0.30 * 0.6 = 0.56 + 0.18 = 0.74
        score = self.calc.compute_score(0.8, 0.6)
        self.assertAlmostEqual(score, 0.74, places=4)

        # Top = 0, Avg = 0
        score_zero = self.calc.compute_score(0.0, 0.0)
        self.assertEqual(score_zero, 0.0)

    def test_calculate_result_dataclass(self):
        res = self.calc.calculate(0.9, 0.8)
        self.assertEqual(res.confidence_level, ConfidenceLevel.HIGH)
        self.assertTrue(res.is_sufficient)
        self.assertAlmostEqual(res.combined_score, 0.87, places=4)

        # Low score result
        res_low = self.calc.calculate(0.3, 0.2)
        self.assertEqual(res_low.confidence_level, ConfidenceLevel.LOW)
        self.assertFalse(res_low.is_sufficient)

    def test_configurable_custom_thresholds(self):
        custom_thresholds = ConfidenceThresholds(
            high=0.85,
            medium=0.60,
            low=0.30,
            weight_top=0.60,
            weight_avg=0.40
        )
        custom_calc = ConfidenceCalculator(thresholds=custom_thresholds)

        # 0.80 was HIGH in default, but is now MEDIUM in custom
        self.assertEqual(custom_calc.classify(0.80), ConfidenceLevel.MEDIUM)
        self.assertEqual(custom_calc.classify(0.85), ConfidenceLevel.HIGH)
        self.assertEqual(custom_calc.classify(0.59), ConfidenceLevel.LOW)
        self.assertEqual(custom_calc.classify(0.29), ConfidenceLevel.NONE)

    def test_invalid_threshold_configurations(self):
        # High must be greater than medium
        with self.assertRaises(ValueError):
            ConfidenceThresholds(high=0.5, medium=0.6, low=0.2)

        # Weights must sum to 1.0
        with self.assertRaises(ValueError):
            ConfidenceThresholds(high=0.7, medium=0.4, low=0.1, weight_top=0.5, weight_avg=0.2)


if __name__ == "__main__":
    unittest.main()
