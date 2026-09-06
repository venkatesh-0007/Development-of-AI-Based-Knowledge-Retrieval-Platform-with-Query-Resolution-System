import unittest
from app.ingestion.chunker import chunk_text

class TestChunker(unittest.TestCase):
    def test_chunk_text_basic(self):
        text = "Sentence one. Sentence two. Sentence three. Sentence four."
        chunks = chunk_text(text, chunk_size=30, overlap=10)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(len(c) <= 35 for c in chunks))

    def test_chunk_text_empty(self):
        self.assertEqual(chunk_text(""), [])
        self.assertEqual(chunk_text("   "), [])

    def test_chunk_text_invalid_params(self):
        with self.assertRaises(ValueError):
            chunk_text("test", chunk_size=0)
        with self.assertRaises(ValueError):
            chunk_text("test", chunk_size=100, overlap=100)
        with self.assertRaises(ValueError):
            chunk_text("test", chunk_size=100, overlap=-1)

if __name__ == "__main__":
    unittest.main()
