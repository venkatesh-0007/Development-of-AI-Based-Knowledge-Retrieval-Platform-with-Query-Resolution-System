"""Automated Voice Processing and Speech Synthesis Unit Tests (Phase 9 & Phase 10).

Validates:
1. Voice payload generation and configuration bounds
2. Transcription text processing and sanitization
3. Error handling: empty transcripts, whitespace, noise, client error tokens
4. Markdown stripping, code block omission, and citation bracket removal for TTS
5. Speech configuration rate/pitch/volume clamping

Note: Real-time browser microphone access and Web Speech API events are client-side
and must be validated manually in a supported browser (Chrome, Edge, Safari 14.1+).
See milestone_4/docs/m4.3-retrieval-prompt-routing-voice-optimization.md for manual test procedures.
"""
import sys
import unittest
from pathlib import Path

m4_root = str(Path(__file__).resolve().parent.parent)
if m4_root not in sys.path:
    sys.path.insert(0, m4_root)

from app.agents.voice import VoiceModule
from app.agents.models import SpeechConfig


class TestVoiceE2E(unittest.TestCase):
    """Automated backend tests for Voice Input and Text-to-Speech Module."""

    def setUp(self):
        self.voice = VoiceModule()

    def test_clean_for_speech_markdown_stripping(self):
        """Markdown headers, bold, bullets, links, and code blocks must be cleanly stripped."""
        markdown = """### Header 3: System Overview
Here is **critical** information with *italic* text and a link to [RFC 793](https://tools.ietf.org/html/rfc793).
- Bullet point 1
- Bullet point 2
```python
def handshake():
    return "SYN-ACK"
```
Inline `variable_name` should be removed.
> Blockquote text here.
---
[1] Citation reference [computer_networks.txt, Chunk 2]
"""
        spoken = self.voice.clean_for_speech(markdown)
        self.assertNotIn("###", spoken)
        self.assertNotIn("**critical**", spoken)
        self.assertIn("critical", spoken)
        self.assertNotIn("https://", spoken)
        self.assertIn("RFC 793", spoken)
        self.assertNotIn("```python", spoken)
        self.assertNotIn("def handshake():", spoken)
        self.assertIn("Code snippet omitted for voice playback", spoken)
        self.assertNotIn("`variable_name`", spoken)
        self.assertNotIn("---", spoken)
        self.assertNotIn("[1]", spoken)
        self.assertNotIn("Chunk 2", spoken)

    def test_clean_for_speech_emoji_removal(self):
        """Visual status emojis should be removed from spoken text to avoid text-to-speech clutter."""
        text = "🟢 HIGH CONFIDENCE 🚀 System operational ⚠️ Warning detected 📊 Data ready"
        spoken = self.voice.clean_for_speech(text)
        for emoji in ["🟢", "🚀", "⚠️", "📊"]:
            self.assertNotIn(emoji, spoken)
        self.assertIn("HIGH CONFIDENCE", spoken)
        self.assertIn("System operational", spoken)

    def test_process_transcription_valid(self):
        """Valid speech transcription is cleaned and approved."""
        ok, text, err = self.voice.process_transcription("  What is TCP three-way handshake?  ")
        self.assertTrue(ok)
        self.assertEqual(text, "What is TCP three-way handshake?")
        self.assertIsNone(err)

    def test_process_transcription_empty_and_whitespace(self):
        """Empty or whitespace-only voice inputs return clear error signals."""
        ok1, _, err1 = self.voice.process_transcription("")
        self.assertFalse(ok1)
        self.assertIn("No speech detected", err1)

        ok2, _, err2 = self.voice.process_transcription("     ")
        self.assertFalse(ok2)
        self.assertIn("No speech detected", err2)

        ok3, _, err3 = self.voice.process_transcription(None)
        self.assertFalse(ok3)
        self.assertIn("empty", err3)

    def test_process_transcription_noise_or_punctuation_only(self):
        """Noise or punctuation without alphanumeric words returns an error."""
        ok, _, err = self.voice.process_transcription("... !!! ??? ---")
        self.assertFalse(ok)
        self.assertIn("recognizable speech", err)

    def test_handle_voice_error_codes(self):
        """Known Web Speech API error codes provide actionable guidance."""
        err_perm = self.voice.handle_voice_error("not-allowed")
        self.assertIn("Microphone permission denied", err_perm["user_message"])

        err_supp = self.voice.handle_voice_error("not-supported")
        self.assertIn("Web Speech API is not supported", err_supp["user_message"])

        err_no_speech = self.voice.handle_voice_error("no-speech")
        self.assertIn("No speech detected", err_no_speech["user_message"])

    def test_build_speech_payload_clamping(self):
        """Speech config parameters (rate, pitch, volume) are safely clamped within Web Speech API limits."""
        custom_cfg = SpeechConfig(rate=5.0, pitch=-1.0, volume=2.5, lang="en-US")
        payload = self.voice.build_speech_payload("Hello World", config_override=custom_cfg)
        self.assertEqual(payload["spoken_text"], "Hello World")
        self.assertEqual(payload["rate"], 2.0)    # Clamped to max 2.0
        self.assertEqual(payload["pitch"], 0.0)   # Clamped to min 0.0
        self.assertEqual(payload["volume"], 1.0)  # Clamped to max 1.0
        self.assertEqual(payload["lang"], "en-US")


if __name__ == "__main__":
    unittest.main()
