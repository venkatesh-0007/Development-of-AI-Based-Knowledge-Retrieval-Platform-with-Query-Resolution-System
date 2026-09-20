"""Unit tests for Milestone 3.3 — Voice Input & Text-to-Speech Module."""
import unittest
from app.agents.models import SpeechConfig
from app.agents.voice import VoiceModule

class TestVoiceModule(unittest.TestCase):
    def setUp(self):
        self.voice = VoiceModule()

    def test_clean_for_speech_markdown_removal(self):
        raw_text = """### Overview of Decision Trees\n\n**Decision Trees** are a *non-parametric* supervised learning method [1].\n\n- Point 1: Easy to interpret\n- Point 2: Handles non-linear data\n\n```python\nmodel = DecisionTreeClassifier()\n```\nFor more details, check [docs](https://scikit-learn.org)."""
        cleaned = self.voice.clean_for_speech(raw_text)

        # Assert no markdown header hashes
        self.assertNotIn("###", cleaned)
        # Assert no code block
        self.assertNotIn("DecisionTreeClassifier()", cleaned)
        self.assertNotIn("```", cleaned)
        # Assert no citation bracket
        self.assertNotIn("[1]", cleaned)
        # Assert no markdown url
        self.assertNotIn("https://", cleaned)
        # Assert bold/italic symbols removed
        self.assertNotIn("**", cleaned)
        self.assertNotIn("*non-parametric*", cleaned)

    def test_clean_for_speech_empty_or_none(self):
        self.assertEqual(self.voice.clean_for_speech(""), "")
        self.assertEqual(self.voice.clean_for_speech(None), "")

    def test_generate_speech_config(self):
        text = "Hello, world! This is a test."
        config = self.voice.generate_speech_config(text, voice_name="en-US-Standard", rate=1.1, pitch=1.0)
        
        self.assertIsInstance(config, SpeechConfig)
        self.assertEqual(config.text, text)
        self.assertEqual(config.voice_name, "en-US-Standard")
        self.assertEqual(config.rate, 1.1)
        self.assertEqual(config.pitch, 1.0)

    def test_sanitize_spoken_turn(self):
        resp = "Grounded Answer: TCP provides reliable delivery [1]. `PORT: 80`."
        spoken = self.voice.clean_for_speech(resp)
        self.assertIn("TCP provides reliable delivery", spoken)
        self.assertNotIn("[1]", spoken)
        self.assertNotIn("`", spoken)

if __name__ == "__main__":
    unittest.main()
