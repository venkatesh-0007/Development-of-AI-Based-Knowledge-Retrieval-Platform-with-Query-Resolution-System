import unittest
from app.ingestion.loaders import extract_text

class TestLoaders(unittest.TestCase):
    def test_txt_loader(self):
        text = extract_text(b"Hello\nWorld from TXT", "test.txt")
        self.assertIn("Hello", text)
        self.assertIn("World from TXT", text)

    def test_csv_loader(self):
        csv_bytes = b"col1,col2\nval1,val2\nval3,val4"
        text = extract_text(csv_bytes, "test.csv")
        self.assertIn("col1", text)
        self.assertIn("val1", text)
        self.assertIn("val4", text)

    def test_unsupported_loader(self):
        with self.assertRaises(ValueError):
            extract_text(b"some content", "test.exe")

if __name__ == "__main__":
    unittest.main()
