"""Automated integrity validation for all benchmark CSV datasets in Milestone 4 and prior milestones.

Verifies:
- UTF-8 encoding compatibility
- Valid, non-empty headers
- Consistent column counts across every row
- Zero unquoted comma delimiter corruption or column-shifting
- Proper RFC 4180 parsing compliance
"""
import os
import csv
import unittest
from pathlib import Path


class TestDatasetIntegrity(unittest.TestCase):
    """Test suite ensuring all CSV datasets conform strictly to CSV standards."""

    def setUp(self):
        self.repo_root = Path(__file__).resolve().parent.parent.parent
        self.m4_docs = self.repo_root / "milestone_4" / "data" / "sample_docs"

    def test_m4_sample_docs_exist(self):
        """Ensure sample_docs directory exists and contains expected datasets."""
        self.assertTrue(self.m4_docs.exists(), f"Directory not found: {self.m4_docs}")
        csv_files = list(self.m4_docs.glob("*.csv"))
        self.assertGreaterEqual(len(csv_files), 3, "Expected at least 3 benchmark CSV files in M4.")

    def test_all_m4_csv_files_integrity(self):
        """Validate every CSV in milestone_4/data/sample_docs for row-column consistency and UTF-8."""
        csv_files = list(self.m4_docs.glob("*.csv"))
        self.assertTrue(len(csv_files) > 0, "No CSV files found in milestone_4/data/sample_docs")

        for csv_path in csv_files:
            with self.subTest(file=csv_path.name):
                # 1. UTF-8 decoding check
                with open(csv_path, "r", encoding="utf-8") as f:
                    content = f.read()
                self.assertGreater(len(content.strip()), 0, f"{csv_path.name} is empty.")

                # 2. Parse using standard RFC 4180 csv reader
                with open(csv_path, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    try:
                        header = next(reader)
                    except StopIteration:
                        self.fail(f"Header missing in {csv_path.name}")

                    self.assertGreater(len(header), 0, f"Empty header in {csv_path.name}")
                    expected_col_count = len(header)

                    # Ensure headers are non-empty strings
                    for col in header:
                        self.assertTrue(bool(col.strip()), f"Blank column header found in {csv_path.name}")

                    # Check every row
                    row_idx = 1
                    for row in reader:
                        row_idx += 1
                        if not row:
                            continue  # Allow trailing empty line if handled by reader
                        self.assertEqual(
                            len(row),
                            expected_col_count,
                            f"Column count mismatch at row {row_idx} of {csv_path.name}: "
                            f"expected {expected_col_count}, got {len(row)}. Content: {row}"
                        )

    def test_cryptography_and_zero_trust_specific_fields(self):
        """Explicitly test the MFA line in cryptography_and_zero_trust.csv for unquoted comma issue."""
        csv_path = self.m4_docs / "cryptography_and_zero_trust.csv"
        self.assertTrue(csv_path.exists())

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            expected_headers = ["Mechanism", "Type", "Key_Length", "Primary_Function", "Security_Property"]
            self.assertEqual(headers, expected_headers)

            rows = list(reader)
            mfa_row = next((r for r in rows if "MFA" in r["Mechanism"]), None)
            self.assertIsNotNone(mfa_row, "MFA row missing from cryptography_and_zero_trust.csv")
            self.assertEqual(mfa_row["Type"], "Access Control")
            self.assertEqual(mfa_row["Security_Property"], "Identity Assurance")
            self.assertIn("know, have, are", mfa_row["Primary_Function"])

    def test_all_repo_csv_files(self):
        """Ensure no malformed CSV exists across the entire project (MS1, MS2, MS3, MS4)."""
        all_csvs = list(self.repo_root.glob("**/*.csv"))
        for csv_path in all_csvs:
            if ".venv" in str(csv_path) or ".git" in str(csv_path):
                continue
            with self.subTest(file=str(csv_path.relative_to(self.repo_root))):
                with open(csv_path, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    header = next(reader)
                    expected_col_count = len(header)
                    for idx, row in enumerate(reader, start=2):
                        if not row:
                            continue
                        self.assertEqual(
                            len(row),
                            expected_col_count,
                            f"Row {idx} in {csv_path} has {len(row)} columns, expected {expected_col_count}"
                        )


if __name__ == "__main__":
    unittest.main()
