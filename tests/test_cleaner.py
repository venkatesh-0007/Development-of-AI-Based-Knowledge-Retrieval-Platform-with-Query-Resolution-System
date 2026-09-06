import unittest
from app.ingestion.cleaner import clean_text

class TestCleaner(unittest.TestCase):
    def test_clean_text_basic(self):
        raw = "  Hello   World!  \r\n\r\n\r\nThis is a test.  "
        cleaned = clean_text(raw)
        self.assertEqual(cleaned, "Hello World!\n\nThis is a test.")

    def test_clean_text_null_bytes(self):
        raw = "Null\x00byte\x00removal"
        self.assertEqual(clean_text(raw), "Null byte removal")

    def test_clean_text_empty(self):
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text("   \n\n  \t  "), "")

if __name__ == "__main__":
    unittest.main()
