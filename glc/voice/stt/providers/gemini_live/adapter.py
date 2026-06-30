"""Gemini Live (streaming voice in via WebSocket) STT provider.

Google's BidiGenerateContent endpoint is full-duplex: the client opens a
WebSocket, sends a `BidiGenerateContentSetup` frame *first*, then streams
audio, and reads transcript chunks back until the server emits
`turnComplete`.

This module has two paths:

* ``_transcribe_via_mock`` — used by the CI test-suite. When
  ``config["mock"]`` is present the adapter talks to the in-repo fake in
  ``tests/voice/stt/mocks/gemini_live_mock.py`` instead of the network.
  This is the path the 7 tests exercise.

* ``_transcribe_via_websocket`` — the real upstream call used outside the
  tests (e.g. for the demo video). CI never runs it.

Both paths obey the same wire rule: the ``setup`` frame is always sent
before the audio frame.
"""

from __future__ import annotations

import base64
import json
import os
import ssl
import time
from typing import Any

from glc.voice.stt.base import STTError, STTProvider, TranscribeResult

# Default Gemini Live model + endpoint. Overridable via ``config``.
# NOTE: only *Live* models expose `bidiGenerateContent`; the plain
# `gemini-3.1-flash-lite` is NOT live-capable.
_DEFAULT_MODEL = "models/gemini-3.1-flash-live-preview"
_WS_ENDPOINT = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)


class Provider(STTProvider):
    name = "gemini_live"

    # ── public entry point ─────────────────────────────────────────
    async def transcribe(self, audio: bytes, mime: str) -> TranscribeResult:
        """Turn an audio clip into text.

        Picks the fake transport when a mock is injected (tests), and the
        real WebSocket transport otherwise.
        """
        mock = self.config.get("mock")
        if mock is not None:
            return await self._transcribe_via_mock(mock, audio, mime)
        return await self._transcribe_via_websocket(audio, mime)

    # ── frame builders (shared by both paths) ──────────────────────
    def _build_setup_frame(self) -> dict[str, Any]:
        """The BidiGenerateContentSetup frame. Must be sent first.

        ``responseModalities`` lives under ``generationConfig`` in the real
        v1beta wire format; the canonical key the tests look for is the
        top-level ``setup`` key, which is preserved.
        """
        model = self.config.get("model", _DEFAULT_MODEL)
        # gemini-3.1-flash-live-preview only supports AUDIO responseModalities.
        # inputAudioTranscription is rejected by the current API (1007 error).
        # Workaround: use outputAudioTranscription to get a text transcript of
        # the model's AUDIO reply. A systemInstruction tells the model to repeat
        # the user's words verbatim, so outputTranscription == input speech.
        modalities = self.config.get("response_modalities", ["AUDIO"])
        return {
            "setup": {
                "model": model,
                "generationConfig": {"responseModalities": modalities},
                "outputAudioTranscription": {},
                "systemInstruction": {
                    "parts": [{
                        "text": (
                            "You are a speech transcription service. "
                            "Repeat back exactly what the user says, "
                            "word for word. Output only the transcription "
                            "with no additional commentary."
                        )
                    }]
                },
            }
        }

    def _build_audio_frame(self, audio: bytes, mime: str) -> dict[str, Any]:
        """The realtimeInput frame carrying the (base64) audio payload.

        Gemini Live requires raw PCM (not a WAV container). If the caller
        passes WAV bytes (detected by the 'RIFF' magic header), the 44-byte
        header is stripped automatically before encoding.
        """
        # Strip WAV container header if present — Gemini Live rejects it.
        _WAV_HEADER_BYTES = 44
        if audio[:4] == b"RIFF":
            audio = audio[_WAV_HEADER_BYTES:]
            mime = "audio/pcm;rate=16000"
        encoded = base64.b64encode(audio).decode("ascii")
        return {
            "realtimeInput": {
                "audio": {"mimeType": mime, "data": encoded},
            }
        }

    # ── mock path (what CI exercises) ──────────────────────────────
    async def _transcribe_via_mock(
        self, mock: Any, audio: bytes, mime: str
    ) -> TranscribeResult:
        """Drive the in-repo fake upstream.

        Order matters: the setup frame is recorded before the audio frame
        so ``frames_sent[0]`` is always the setup frame (the Live API
        rejects sessions where audio arrives first).
        """
        mock.record_frame(self._build_setup_frame())
        mock.record_frame(self._build_audio_frame(audio, mime))
        # The fake returns a canned TranscribeResult or raises STTError;
        # let the error propagate untouched.
        return await mock.transcribe(audio, mime)

    # ── real path (for the demo; not run by CI) ────────────────────
    async def _transcribe_via_websocket(
        self, audio: bytes, mime: str
    ) -> TranscribeResult:
        """Real Gemini Live call over the BidiGenerateContent WebSocket.

        Flow (per https://ai.google.dev/api/multimodal-live):

          1. Open ``{_WS_ENDPOINT}?key=$GEMINI_API_KEY``.
          2. Send the setup frame FIRST (includes outputAudioTranscription
             and systemInstruction to transcribe the reply verbatim).
          3. Send the audio as a ``realtimeInput.audio`` frame with raw PCM
             at 16 kHz (WAV header stripped if present), then signal
             ``audioStreamEnd`` so the server closes the input turn.
          4. Read messages, accumulating text from
             ``serverContent.outputTranscription.text`` (primary), falling
             back to ``inputTranscription`` or ``modelTurn.parts[].text``.
             Ignore ``setupComplete`` / ``usageMetadata`` frames. Stop on
             ``serverContent.turnComplete``.
          5. Wrap any failure in ``STTError``.

        Requires ``GEMINI_API_KEY`` in the environment (or ``config``).
        """
        try:
            import websockets
        except ImportError as exc:  # pragma: no cover - dependency present
            raise STTError("the 'websockets' package is required", status=None) from exc

        api_key = self.config.get("api_key") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise STTError("GEMINI_API_KEY is not set", status=None)

        url = f"{_WS_ENDPOINT}?key={api_key}"
        start = time.monotonic()
        transcript: list[str] = []

        # Allow disabling SSL verification for corporate proxies that inject
        # non-standard CA certificates (set config["ssl_verify"] = False).
        ssl_ctx: ssl.SSLContext | bool = True
        if self.config.get("ssl_verify") is False:
            ssl_ctx = ssl.create_default_context()
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = ssl.CERT_NONE

        try:
            async with websockets.connect(url, max_size=None, ssl=ssl_ctx) as ws:
                # 1. setup must be the first frame
                await ws.send(json.dumps(self._build_setup_frame()))
                # 2. push the audio, then close the input turn
                await ws.send(json.dumps(self._build_audio_frame(audio, mime)))
                await ws.send(json.dumps({"realtimeInput": {"audioStreamEnd": True}}))
                # 3. drain responses until the turn completes.
                # Lock onto the first field that produces text and ignore all
                # others for the rest of the session — prevents the same
                # sentence appearing twice when both outputTranscription and
                # modelTurn fire in separate messages.
                preferred_source: str | None = None
                async for raw in ws:
                    data = json.loads(raw)
                    server_content = data.get("serverContent")
                    if not server_content:
                        # Skip non-content frames: setupComplete, usageMetadata,
                        # sessionResumptionUpdate, etc.
                        continue
                    output_tx = server_content.get("outputTranscription")
                    input_tx = server_content.get("inputTranscription")
                    model_turn = server_content.get("modelTurn")

                    if output_tx and output_tx.get("text"):
                        preferred_source = preferred_source or "outputTranscription"
                        if preferred_source == "outputTranscription":
                            transcript.append(output_tx["text"])
                    elif input_tx and input_tx.get("text"):
                        preferred_source = preferred_source or "inputTranscription"
                        if preferred_source == "inputTranscription":
                            transcript.append(input_tx["text"])
                    elif model_turn and preferred_source is None:
                        for part in model_turn.get("parts", []):
                            text = part.get("text")
                            if text:
                                transcript.append(text)
                        if transcript:
                            preferred_source = "modelTurn"

                    if server_content.get("turnComplete"):
                        break
        except STTError:
            raise
        except Exception as exc:  # noqa: BLE001 - surface any upstream failure
            raise STTError(f"Gemini Live WebSocket error: {exc}", status=None) from exc

        duration_ms = int((time.monotonic() - start) * 1000)
        return TranscribeResult(
            text="".join(transcript),
            language=self.config.get("language", "en"),
            duration_ms=duration_ms,
            provider=self.name,
            cost_usd=0.0,
        )
