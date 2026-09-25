"""Voice Input and Text-to-Speech Module for Milestone 4 (M4.3).

Provides speech synthesis text preparation, markdown sanitization for natural audio output,
and Web Speech API configuration helpers.
"""
import re
import logging
from typing import Dict, Any, Optional
from .models import SpeechConfig

logger = logging.getLogger(__name__)

class VoiceModule:
    """Handles text-to-speech sanitization, utterance formatting, and Web Speech API payload preparation."""

    def __init__(self, default_config: Optional[SpeechConfig] = None):
        self.config = default_config or SpeechConfig()

    def clean_for_speech(self, text: str) -> str:
        """
        Sanitizes raw grounded markdown text for natural, fluent text-to-speech audio rendering.
        Strips markdown hashes, code blocks, bullet points, asterisks, URLs, and citation brackets.
        """
        if not text:
            return ""

        cleaned = text

        # 1. Remove fenced code blocks and inline code
        cleaned = re.sub(r"```[\s\S]*?```", " Code snippet omitted for voice playback. ", cleaned)
        cleaned = re.sub(r"`[^`]+`", "", cleaned)

        # 2. Remove markdown links [text](url) -> text
        cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)

        # 3. Remove standalone citation indicators e.g. [1], [tcp_networking.txt, Chunk 1]
        cleaned = re.sub(r"\[(?:Citation\s*)?[^\]]+\]", "", cleaned)

        # 4. Remove markdown headers, bold, italics, blockquotes, horizontal rules
        cleaned = re.sub(r"^[#\s]+", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
        cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)
        cleaned = re.sub(r"^>\s*", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"^[-*•]\s*", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"^-{3,}$", "", cleaned, flags=re.MULTILINE)

        # 5. Remove emojis and non-standard symbols for smoother voice reading
        cleaned = re.sub(r"[🟢🟡🟠🔴⚪🚀🤖🧠💬📚📝🔍👉📌⚠️🎙️📊]", "", cleaned)

        # 6. Normalize whitespace and punctuation
        cleaned = re.sub(r"\n+", ". ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = re.sub(r"\.\s*\.", ".", cleaned)

        return cleaned

    def generate_speech_config(
        self,
        rate: float = 1.0,
        pitch: float = 1.0,
        volume: float = 1.0,
        lang: str = "en-US"
    ) -> SpeechConfig:
        return SpeechConfig(rate=rate, pitch=pitch, volume=volume, lang=lang)

    def build_speech_payload(
        self,
        text: str,
        config_override: Optional[SpeechConfig] = None
    ) -> Dict[str, Any]:
        cfg = config_override or self.config
        spoken_text = self.clean_for_speech(text)
        return {
            "spoken_text": spoken_text,
            "raw_text_length": len(text),
            "spoken_text_length": len(spoken_text),
            "lang": cfg.lang,
            "rate": max(0.5, min(2.0, cfg.rate)),
            "pitch": max(0.0, min(2.0, cfg.pitch)),
            "volume": max(0.0, min(1.0, cfg.volume))
        }
