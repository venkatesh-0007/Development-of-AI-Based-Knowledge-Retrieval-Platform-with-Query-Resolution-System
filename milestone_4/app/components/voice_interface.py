"""Web Speech API Component for Streamlit (M4.3 Voice Reliability & TTS).

Provides embedded browser speech recognition (Microphone STT) and
Speech Synthesis (TTS) audio player with start, pause, resume, and stop controls.
"""
import json
import streamlit as st
import streamlit.components.v1 as components
from typing import Optional

def render_voice_input_widget(element_id: str = "voice_stt_main") -> Optional[str]:
    """
    Renders an HTML5/JavaScript Web Speech API microphone widget.
    Captures spoken audio in real-time and communicates via session state or text input.
    """
    html_code = f"""
    <div id="voice-stt-container-{element_id}" style="
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: #1e293b;
        padding: 10px 16px;
        border-radius: 10px;
        border: 1px solid #334155;
        color: #f1f5f9;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 12px;
    ">
        <button id="mic-btn-{element_id}" style="
            background: #2563eb;
            color: #ffffff;
            border: none;
            padding: 8px 14px;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 6px;
            transition: all 0.2s ease;
        " onclick="toggleRecognition_{element_id}()">
            <span id="mic-icon-{element_id}">🎙️</span>
            <span id="mic-label-{element_id}">Voice Query</span>
        </button>

        <div id="stt-status-{element_id}" style="
            font-size: 13px;
            color: #94a3b8;
            flex-grow: 1;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        ">
            Web Speech STT: Click to dictate query...
        </div>
    </div>

    <script>
    (function() {{
        let recognition = null;
        let isListening = false;
        const btn = document.getElementById("mic-btn-{element_id}");
        const label = document.getElementById("mic-label-{element_id}");
        const status = document.getElementById("stt-status-{element_id}");

        const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRec) {{
            status.innerText = "Web Speech API not supported in this browser.";
            btn.disabled = true;
            btn.style.opacity = "0.5";
            return;
        }}

        recognition = new SpeechRec();
        recognition.continuous = false;
        recognition.interimResults = true;
        recognition.lang = "en-US";

        recognition.onstart = function() {{
            isListening = true;
            btn.style.background = "#dc2626";
            label.innerText = "Listening...";
            status.innerText = "Speak now into microphone...";
        }};

        recognition.onresult = function(event) {{
            let interim = "";
            let finalTranscript = "";
            for (let i = event.resultIndex; i < event.results.length; ++i) {{
                if (event.results[i].isFinal) {{
                    finalTranscript += event.results[i][0].transcript;
                }} else {{
                    interim += event.results[i][0].transcript;
                }}
            }}
            status.innerText = finalTranscript || interim || "Listening...";
            if (finalTranscript) {{
                // Auto-fill chat input if present
                const chatInputs = window.parent.document.querySelectorAll("textarea[data-testid='stChatInputTextArea'], input[type='text']");
                if (chatInputs.length > 0) {{
                    chatInputs[0].value = finalTranscript;
                    chatInputs[0].dispatchEvent(new Event('input', {{ bubbles: true }}));
                }}
            }}
        }};

        recognition.onerror = function(event) {{
            status.innerText = "Speech error: " + event.error;
            stop_{element_id}();
        }};

        recognition.onend = function() {{
            stop_{element_id}();
        }};

        function stop_{element_id}() {{
            isListening = false;
            btn.style.background = "#2563eb";
            label.innerText = "Voice Query";
        }}

        window.toggleRecognition_{element_id} = function() {{
            if (!recognition) return;
            if (isListening) {{
                recognition.stop();
            }} else {{
                try {{
                    recognition.start();
                }} catch (e) {{
                    status.innerText = "Mic error: " + e.message;
                }}
            }}
        }};
    }})();
    </script>
    """
    components.html(html_code, height=55)
    return None

def render_tts_player(text: str, element_id: str = "tts_player") -> None:
    """
    Renders an HTML5 speech synthesis audio control bar (Play, Pause, Resume, Stop).
    """
    if not text:
        return

    escaped = json.dumps(text)
    html_code = f"""
    <div id="tts-container-{element_id}" style="
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: #0f172a;
        padding: 8px 14px;
        border-radius: 8px;
        border: 1px solid #1e293b;
        color: #e2e8f0;
        margin-top: 8px;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-size: 12px;
    ">
        <span style="color: #38bdf8; font-weight: 600;">🔊 Voice TTS:</span>
        <button id="btn-play-{element_id}" style="background:#0284c7;color:#fff;border:none;padding:4px 8px;border-radius:4px;cursor:pointer;" onclick="speak_{element_id}()">▶ Play</button>
        <button id="btn-pause-{element_id}" style="background:#475569;color:#fff;border:none;padding:4px 8px;border-radius:4px;cursor:pointer;" onclick="pause_{element_id}()">⏸ Pause</button>
        <button id="btn-stop-{element_id}" style="background:#ef4444;color:#fff;border:none;padding:4px 8px;border-radius:4px;cursor:pointer;" onclick="stop_{element_id}()">⏹ Stop</button>
        <span id="tts-status-{element_id}" style="color:#94a3b8;margin-left:6px;">Ready</span>
    </div>

    <script>
    (function() {{
        const textToSpeak = {escaped};
        let currentUtterance = null;
        const status = document.getElementById("tts-status-{element_id}");

        window.speak_{element_id} = function() {{
            if (!('speechSynthesis' in window)) {{
                status.innerText = "TTS not supported.";
                return;
            }}
            window.speechSynthesis.cancel();
            currentUtterance = new SpeechSynthesisUtterance(textToSpeak);
            currentUtterance.rate = 1.0;
            currentUtterance.pitch = 1.0;
            currentUtterance.lang = "en-US";

            currentUtterance.onstart = function() {{
                status.innerText = "Playing...";
            }};
            currentUtterance.onend = function() {{
                status.innerText = "Finished";
            }};
            currentUtterance.onerror = function() {{
                status.innerText = "Error";
            }};

            window.speechSynthesis.speak(currentUtterance);
        }};

        window.pause_{element_id} = function() {{
            if ('speechSynthesis' in window && window.speechSynthesis.speaking) {{
                if (window.speechSynthesis.paused) {{
                    window.speechSynthesis.resume();
                    status.innerText = "Resumed";
                }} else {{
                    window.speechSynthesis.pause();
                    status.innerText = "Paused";
                }}
            }}
        }};

        window.stop_{element_id} = function() {{
            if ('speechSynthesis' in window) {{
                window.speechSynthesis.cancel();
                status.innerText = "Stopped";
            }}
        }};
    }})();
    </script>
    """
    components.html(html_code, height=48)
