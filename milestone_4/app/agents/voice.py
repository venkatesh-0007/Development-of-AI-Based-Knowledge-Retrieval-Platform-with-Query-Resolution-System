"""Voice Input and Text-to-Speech Module for Milestone 4 (M4.3, Phase 10, & Phase 17).

Features:
- Speech synthesis text preparation with markdown and citation stripping
- Automated transcription text validation, normalization, and sanitization
- Structured error handling for Web Speech API edge cases:
  * Unsupported browser
  * Permission denied / microphone blocked
  * No speech detected
  * Audio capture failure
  * Network disconnection
- Clean spoken-text formatting for browser TTS
"""
import re
import logging
from typing import Dict, Any, Optional, Tuple
from .models import SpeechConfig

logger = logging.getLogger(__name__)


class VoiceModule:
    """Handles text-to-speech sanitization, utterance formatting, and Web Speech API payload preparation."""

    ERROR_GUIDANCE = {
        "not-allowed": "Microphone permission denied. Please allow microphone access in your browser settings.",
        "permission-denied": "Microphone permission was blocked. Check browser address bar to grant audio access.",
        "no-speech": "No speech detected. Please speak clearly into your microphone and try again.",
        "audio-capture": "No microphone hardware detected or audio capture failed. Ensure a working microphone is connected.",
        "network": "Speech recognition network error occurred. Please verify your internet connection.",
        "not-supported": "Web Speech API is not supported in this browser. Recommended browsers: Chrome, Edge, Safari 14.1+.",
        "aborted": "Speech listening was aborted before transcription completed.",
        "language-not-supported": "Selected speech recognition language is not supported by your browser engine."
    }

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

        # 2. Remove markdown links [text](url) -> text, and bare URLs
        cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
        cleaned = re.sub(r"https?://\S+", "", cleaned)

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
        cleaned = re.sub(r"[🟢🟡🟠🔴⚪🚀🤖🧠💬📚📝🔍👉📌⚠️🎙️📊🧩📐]", "", cleaned)

        # 6. Normalize whitespace and punctuation
        cleaned = re.sub(r"\n+", ". ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = re.sub(r"\.\s*\.", ".", cleaned)

        return cleaned

    def process_transcription(self, raw_transcript: Optional[str]) -> Tuple[bool, str, Optional[str]]:
        """
        Validates, normalizes, and sanitizes incoming voice transcription text.
        Returns: (is_valid: bool, sanitized_text: str, error_message: Optional[str])
        """
        if raw_transcript is None:
            return False, "", "Transcription payload is empty (null)."

        cleaned = raw_transcript.strip()
        if not cleaned:
            return False, "", "No speech detected (empty transcript)."

        # Check for client-reported error tokens
        if cleaned.startswith("[voice_error:") and cleaned.endswith("]"):
            err_code = cleaned[13:-1].strip()
            guidance = self.handle_voice_error(err_code)
            return False, "", guidance.get("user_message", f"Speech recognition error: {err_code}")

        # Check for punctuation-only / noise
        alphanumeric = re.findall(r"[a-zA-Z0-9]", cleaned)
        if not alphanumeric:
            return False, "", "Audio did not contain recognizable speech words."

        # Normalize whitespace
        sanitized = re.sub(r"\s+", " ", cleaned)
        return True, sanitized, None

    def handle_voice_error(self, error_code: str, details: Optional[str] = None) -> Dict[str, str]:
        """Provides user-friendly recovery instructions for Web Speech API error states."""
        clean_code = str(error_code).lower().strip()
        message = self.ERROR_GUIDANCE.get(clean_code, f"Voice interaction error ({clean_code}). Please retry or type query.")
        return {
            "error_code": clean_code,
            "user_message": message,
            "details": details or "",
            "recovery_action": "Try typing your question or grant microphone permissions in browser settings."
        }

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
