"""Unit tests for Milestone 3.4 — Response Transparency Panel."""
import unittest
from app.agents.models import (
    RetrievalChunk,
    SourceAttribution,
    TransparencyScoreBreakdown,
    TransparencyPanelPayload,
    ConfidenceLevel
)

class TestResponseTransparency(unittest.TestCase):
    def test_transparency_score_breakdown_high_confidence(self):
        chunks = [
            RetrievalChunk(chunk_id="c1", document_name="doc_tcp.txt", content="TCP handshake 3-way.", score=0.92, rank=1),
            RetrievalChunk(chunk_id="c2", document_name="doc_tcp.txt", content="TCP congestion control.", score=0.88, rank=2)
        ]
        breakdown = TransparencyScoreBreakdown.from_chunks(chunks)
        
        self.assertEqual(breakdown.top_chunk_score, 0.92)
        self.assertEqual(breakdown.avg_top_k_score, 0.90)
        self.assertEqual(breakdown.top_k_count, 2)
        # combined: 0.7 * 0.92 + 0.3 * 0.90 = 0.644 + 0.27 = 0.914
        self.assertAlmostEqual(breakdown.combined_confidence, 0.914, places=3)
        self.assertEqual(breakdown.confidence_level, ConfidenceLevel.HIGH)
        self.assertEqual(len(breakdown.low_confidence_reasons), 0)

    def test_transparency_score_breakdown_low_confidence(self):
        chunks = [
            RetrievalChunk(chunk_id="c1", document_name="doc_misc.txt", content="Vague mention.", score=0.40, rank=1)
        ]
        breakdown = TransparencyScoreBreakdown.from_chunks(chunks, threshold=0.35)
        
        self.assertEqual(breakdown.confidence_level, ConfidenceLevel.LOW)
        self.assertGreater(len(breakdown.low_confidence_reasons), 0)
        self.assertIn("Top chunk relevance is below 0.50", breakdown.low_confidence_reasons[0])

    def test_transparency_payload_assembly(self):
        chunks = [
            RetrievalChunk(chunk_id="c1", document_name="doc_tcp.txt", content="TCP handshake.", score=0.85, rank=1)
        ]
        breakdown = TransparencyScoreBreakdown.from_chunks(chunks)
        sources = [
            SourceAttribution(document_name="doc_tcp.txt", chunk_id="c1", relevance_score=0.85, snippet="TCP handshake.")
        ]

        payload = TransparencyPanelPayload(
            retrieved_chunks=chunks,
            citations=sources,
            score_breakdown=breakdown
        )

        self.assertEqual(len(payload.retrieved_chunks), 1)
        self.assertEqual(len(payload.citations), 1)
        self.assertEqual(payload.score_breakdown.top_chunk_score, 0.85)

if __name__ == "__main__":
    unittest.main()
