"""Web Speech API Component for Streamlit (M3.3).

Provides embedded browser speech recognition (Microphone STT) and
Speech Synthesis (TTS) audio player with start, pause, resume, and stop controls.
"""
import json
import streamlit.components.v1 as components
from typing import Optional

def render_voice_input_widget(key: str = "voice_input_widget") -> None:
    """
    Renders an HTML5/JavaScript Web Speech API microphone widget.
    Captures spoken audio in real-time and populates the query input.
    """
    html_code = """
    <div id="voice-stt-container" style="
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        background: #1e222a;
        padding: 12px 18px;
        border-radius: 12px;
        border: 1px solid #333a46;
        color: #e6edf3;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 14px;
    ">
        <button id="mic-toggle-btn" style="
            background: #238636;
            color: #ffffff;
            border: none;
            padding: 10px 16px;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: all 0.2s ease;
        " onclick="toggleSpeechRecognition()">
            <span id="mic-icon">🎤</span>
            <span id="mic-label">Start Voice Query</span>
        </button>

        <div id="stt-status-indicator" style="
            font-size: 13px;
            color: #8b949e;
            flex-grow: 1;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        ">
            Click mic to speak query via Web Speech API...
        </div>

        <button id="mic-clear-btn" style="
            background: #21262d;
            color: #c9d1d9;
            border: 1px solid #30363d;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 12px;
            cursor: pointer;
        " onclick="clearVoiceTranscript()">
            Clear
        </button>
    </div>

    <style>
        .listening-pulse {
            animation: pulse-red 1.5s infinite;
            background: #da3633 !important;
        }
        @keyframes pulse-red {
            0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(218, 54, 51, 0.7); }
            70% { transform: scale(1.03); box-shadow: 0 0 0 10px rgba(218, 54, 51, 0); }
            100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(218, 54, 51, 0); }
        }
    </style>

    <script>
        let recognition = null;
        let isListening = false;

        function initSpeechRecognition() {
            window.SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            if (!window.SpeechRecognition) {
                document.getElementById("stt-status-indicator").innerHTML = "<span style='color:#f85149;'>⚠️ Web Speech API not supported in this browser. Please use Chrome, Edge, or Safari.</span>";
                document.getElementById("mic-toggle-btn").disabled = true;
                return false;
            }

            recognition = new window.SpeechRecognition();
            recognition.continuous = false;
            recognition.interimResults = true;
            recognition.lang = 'en-US';

            recognition.onstart = function() {
                isListening = true;
                const btn = document.getElementById("mic-toggle-btn");
                btn.classList.add("listening-pulse");
                document.getElementById("mic-label").innerText = "Listening...";
                document.getElementById("stt-status-indicator").innerText = "🎙️ Listening to your voice... Speak your query clearly.";
            };

            recognition.onresult = function(event) {
                let transcript = "";
                for (let i = event.resultIndex; i < event.results.length; ++i) {
                    transcript += event.results[i][0].transcript;
                }
                document.getElementById("stt-status-indicator").innerHTML = "<strong>Transcribed:</strong> <em>" + transcript + "</em>";
                
                // Copy to parent window's Streamlit text input if accessible
                try {
                    const inputs = window.parent.document.querySelectorAll("input[type='text'], textarea");
                    for (let inp of inputs) {
                        if (inp.getAttribute("aria-label") && inp.getAttribute("aria-label").includes("query") || inp.placeholder.includes("query")) {
                            inp.value = transcript;
                            inp.dispatchEvent(new Event('input', { bubbles: true }));
                            break;
                        }
                    }
                } catch(e) {}
            };

            recognition.onerror = function(event) {
                isListening = false;
                resetMicButton();
                document.getElementById("stt-status-indicator").innerHTML = "<span style='color:#f85149;'>Speech recognition error: " + event.error + "</span>";
            };

            recognition.onend = function() {
                isListening = false;
                resetMicButton();
            };

            return true;
        }

        function toggleSpeechRecognition() {
            if (!recognition) {
                if (!initSpeechRecognition()) return;
            }

            if (isListening) {
                recognition.stop();
            } else {
                try {
                    recognition.start();
                } catch(e) {
                    initSpeechRecognition();
                    recognition.start();
                }
            }
        }

        function resetMicButton() {
            const btn = document.getElementById("mic-toggle-btn");
            btn.classList.remove("listening-pulse");
            document.getElementById("mic-label").innerText = "Start Voice Query";
        }

        function clearVoiceTranscript() {
            if (isListening && recognition) recognition.stop();
            document.getElementById("stt-status-indicator").innerText = "Transcript cleared. Click mic to speak...";
        }

        // Initialize on load
        initSpeechRecognition();
    </script>
    """
    components.html(html_code, height=75)

def render_tts_player(spoken_text: str, element_key: str = "tts_player") -> None:
    """
    Renders an inline Text-to-Speech audio control player (Play, Pause, Resume, Stop).
    """
    sanitized_json = json.dumps(spoken_text)
    html_code = f"""
    <div id="tts-container-{element_key}" style="
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 8px 14px;
        margin-top: 10px;
        display: flex;
        align-items: center;
        gap: 10px;
    ">
        <span style="font-size: 13px; font-weight: 600; color: #58a6ff; display: flex; align-items: center; gap: 4px;">
            🔊 Spoken Audio (TTS):
        </span>
        
        <button id="tts-play-{element_key}" onclick="playTTS_{element_key}()" style="
            background: #238636; color: white; border: none; padding: 5px 12px; border-radius: 6px; font-size: 12px; font-weight: 600; cursor: pointer;
        ">▶ Play</button>

        <button id="tts-pause-{element_key}" onclick="pauseTTS_{element_key}()" style="
            background: #d29922; color: white; border: none; padding: 5px 12px; border-radius: 6px; font-size: 12px; font-weight: 600; cursor: pointer;
        ">⏸ Pause</button>

        <button id="tts-stop-{element_key}" onclick="stopTTS_{element_key}()" style="
            background: #da3633; color: white; border: none; padding: 5px 12px; border-radius: 6px; font-size: 12px; font-weight: 600; cursor: pointer;
        ">⏹ Stop</button>

        <span id="tts-status-{element_key}" style="font-size: 12px; color: #8b949e; margin-left: auto;">
            Ready to read response aloud
        </span>
    </div>

    <script>
        const textToSpeak_{element_key} = {sanitized_json};
        let utterance_{element_key} = null;

        function playTTS_{element_key}() {{
            if (!('speechSynthesis' in window)) {{
                document.getElementById('tts-status-{element_key}').innerText = 'TTS not supported in browser';
                return;
            }}

            if (window.speechSynthesis.paused) {{
                window.speechSynthesis.resume();
                document.getElementById('tts-status-{element_key}').innerText = 'Playing...';
                return;
            }}

            window.speechSynthesis.cancel();
            utterance_{element_key} = new SpeechSynthesisUtterance(textToSpeak_{element_key});
            utterance_{element_key}.lang = 'en-US';
            utterance_{element_key}.rate = 1.0;
            utterance_{element_key}.pitch = 1.0;

            utterance_{element_key}.onstart = function() {{
                document.getElementById('tts-status-{element_key}').innerText = '🔊 Reading response...';
            }};
            utterance_{element_key}.onend = function() {{
                document.getElementById('tts-status-{element_key}').innerText = 'Completed playback';
            }};
            utterance_{element_key}.onerror = function(e) {{
                document.getElementById('tts-status-{element_key}').innerText = 'Audio playback stopped';
            }};

            window.speechSynthesis.speak(utterance_{element_key});
        }}

        function pauseTTS_{element_key}() {{
            if ('speechSynthesis' in window && window.speechSynthesis.speaking) {{
                window.speechSynthesis.pause();
                document.getElementById('tts-status-{element_key}').innerText = 'Paused';
            }}
        }}

        function stopTTS_{element_key}() {{
            if ('speechSynthesis' in window) {{
                window.speechSynthesis.cancel();
                document.getElementById('tts-status-{element_key}').innerText = 'Stopped';
            }}
        }}
    </script>
    """
    components.html(html_code, height=55)
