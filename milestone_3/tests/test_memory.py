"""Unit tests for Milestone 3.2 — Conversation Memory Agent."""
import unittest
from app.agents.models import (
    ConversationSession,
    ConversationTurn,
    TransparencyPanelPayload
)
from app.agents.memory import ConversationMemoryAgent

class TestConversationMemoryAgent(unittest.TestCase):
    def setUp(self):
        self.memory_agent = ConversationMemoryAgent(max_history_turns=5)
        self.session = self.memory_agent.get_or_create_session("test_session_1")

    def test_session_creation_and_retrieval(self):
        session = self.memory_agent.get_or_create_session("session_abc")
        self.assertEqual(session.session_id, "session_abc")
        self.assertEqual(len(session.turns), 0)

        # Same session retrieved
        session2 = self.memory_agent.get_or_create_session("session_abc")
        self.assertIs(session, session2)

    def test_record_turn_and_history_pruning(self):
        agent = ConversationMemoryAgent(max_history_turns=3)
        session = agent.get_or_create_session("prune_session")

        for i in range(5):
            turn = ConversationTurn(
                query=f"Query {i}",
                effective_query=f"Effective Query {i}",
                response=f"Response {i}",
                confidence_score=0.9,
                sources=[],
                key_entities=[f"Entity_{i}"]
            )
            agent.record_turn(session, turn)

        self.assertEqual(len(session.turns), 3)
        self.assertEqual(session.turns[0].query, "Query 2")
        self.assertEqual(session.turns[-1].query, "Query 4")

    def test_anaphora_resolution_pronoun_it(self):
        # Turn 1: User asks about TCP
        turn1 = ConversationTurn(
            query="What is Transmission Control Protocol (TCP)?",
            effective_query="What is Transmission Control Protocol (TCP)?",
            response="TCP is a connection-oriented transport layer protocol.",
            confidence_score=0.95,
            sources=[],
            key_entities=["Transmission Control Protocol (TCP)"]
        )
        self.memory_agent.record_turn(self.session, turn1)

        # Turn 2: Follow up with "How does it work?"
        raw_followup = "How does it work?"
        rewritten = self.memory_agent.contextualize_query(raw_followup, self.session)
        
        self.assertIn("Transmission Control Protocol (TCP)", rewritten)
        self.assertTrue(rewritten.lower().startswith("how does transmission control protocol (tcp)"))

    def test_anaphora_resolution_pronoun_its(self):
        # Turn 1: User asks about Random Forest
        turn1 = ConversationTurn(
            query="Tell me about Random Forest in machine learning.",
            effective_query="Tell me about Random Forest in machine learning.",
            response="Random Forest is an ensemble learning method.",
            confidence_score=0.92,
            sources=[],
            key_entities=["Random Forest"]
        )
        self.memory_agent.record_turn(self.session, turn1)

        # Turn 2: Follow up with "What are its key advantages?"
        raw_followup = "What are its key advantages?"
        rewritten = self.memory_agent.contextualize_query(raw_followup, self.session)
        
        self.assertIn("Random Forest", rewritten)
        self.assertIn("key advantages", rewritten.lower())

    def test_context_switching_no_pronoun(self):
        # Turn 1: User asks about Support Vector Machines
        turn1 = ConversationTurn(
            query="What is Support Vector Machines (SVM)?",
            effective_query="What is Support Vector Machines (SVM)?",
            response="SVM is a supervised learning model.",
            confidence_score=0.90,
            sources=[],
            key_entities=["Support Vector Machines (SVM)"]
        )
        self.memory_agent.record_turn(self.session, turn1)

        # Turn 2: Topic switch to DNS without pronouns
        new_query = "What port does DNS use?"
        rewritten = self.memory_agent.contextualize_query(new_query, self.session)
        
        # Should not inject SVM into DNS query
        self.assertEqual(rewritten, new_query)

    def test_clear_session(self):
        turn1 = ConversationTurn(
            query="What is CNN?",
            effective_query="What is CNN?",
            response="Convolutional Neural Network",
            confidence_score=0.9,
            sources=[]
        )
        self.memory_agent.record_turn(self.session, turn1)
        self.assertEqual(len(self.session.turns), 1)

        self.memory_agent.clear_session("test_session_1")
        reloaded = self.memory_agent.get_or_create_session("test_session_1")
        self.assertEqual(len(reloaded.turns), 0)

if __name__ == "__main__":
    unittest.main()
