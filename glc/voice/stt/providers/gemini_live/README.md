# Gemini Live STT Provider

Real-time speech-to-text using Google's
[BidiGenerateContent WebSocket endpoint](https://ai.google.dev/api/multimodal-live).
Implemented for EAG Session 11 (Group G4 — Gemini Live STT).

---

## Architecture

```
  Microphone / audio file
         │  raw 16-bit PCM  (audio/pcm;rate=16000)
         ▼
  Provider.transcribe()          adapter.py
         │
         ├─ mock in config? ──► GeminiLiveMock   (CI / unit tests)
         │
         └─ otherwise:
              │
              ▼
         Open WSS connection to Gemini Live API
              │
              ├─ 1. Send setup frame  ◄── MUST be first
              │       model + generationConfig.responseModalities
              │       + inputAudioTranscription
              │
              ├─ 2. Send audio frame
              │       realtimeInput.audio  (base64 PCM)
              │
              ├─ 3. Send audioStreamEnd
              │
              └─ 4. Read responses until turnComplete
                      serverContent.inputTranscription.text
                      (fallback: modelTurn.parts[].text)
                           │
                           ▼
                    TranscribeResult
                    (text, language, duration_ms,
                     provider, cost_usd)
```

### Key files

| File | Purpose |
|---|---|
| `adapter.py` | `Provider` class — two paths: mock (CI) and real WebSocket |
| `schemas.py` | Pydantic config schema for this provider |
| `tests/voice/stt/test_gemini_live.py` | 7 tests (6 structural + 1 behavioural) |
| `tests/voice/stt/mocks/gemini_live_mock.py` | In-process fake that replaces the real WebSocket |

### Wire protocol summary

1. Open `wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key=<KEY>`
2. Send `BidiGenerateContentSetup` **first** (model + `generationConfig` + `inputAudioTranscription`)
3. Send `realtimeInput` with base64-encoded raw PCM audio
4. Send `realtimeInput.audioStreamEnd = true` to close the input turn
5. Read `serverContent.inputTranscription.text` chunks until `serverContent.turnComplete`

---

## Issues Hit and How They Were Fixed

### 1. SSL certificate verification failure

**Error:**
```
[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed:
Basic Constraints of CA cert not marked critical
```

**Cause:** Corporate proxy (Infoblox) intercepts TLS and injects its own CA
certificate, which does not meet strict standard requirements (`Basic Constraints`
must be marked critical).

**Fix:** Added `ssl_verify` config option to `_transcribe_via_websocket`. When
`config["ssl_verify"] is False`, the adapter creates an `ssl.SSLContext` with
`check_hostname = False` and `verify_mode = ssl.CERT_NONE` before connecting:

```python
ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE
async with websockets.connect(url, max_size=None, ssl=ssl_ctx) as ws:
```

---

### 2. Wrong model name (`gemini-2.0-flash-live-001`)

**Error:**
```
received 1008 (policy violation) models/gemini-2.0-flash-live-001
is not found for API version v1beta, or is not supported for
bidiGenerateContent.
```

**Cause:** `gemini-2.0-flash-live-001` has been shut down. Only Live-capable
models expose the `bidiGenerateContent` endpoint.

**Fix:** The correct model for this provider is `gemini-3.1-flash-live-preview`,
which is the default in the adapter:

```python
_DEFAULT_MODEL = "models/gemini-3.1-flash-live-preview"
```

---

### 3. `responseModalities` in wrong location in setup frame

**Error:**
```
received 1007 (invalid frame payload data) Invalid JSON payload received.
Unknown name "responseModalities" at 'setup': Cannot find field.
```

**Cause:** `responseModalities` must be nested inside `generationConfig` in the
raw v1beta WebSocket wire format — placing it directly under `setup` is invalid
even though some SDK documentation examples show it at the top level (the SDK
handles that translation internally).

**Fix:** Keep `responseModalities` under `generationConfig`:

```json
{
  "setup": {
    "model": "models/gemini-3.1-flash-live-preview",
    "generationConfig": { "responseModalities": ["AUDIO"] },
    "inputAudioTranscription": {}
  }
}
```

---

### 4. Sending WAV instead of raw PCM

**Cause:** The Gemini Live API requires raw 16-bit little-endian PCM, not
a WAV container. Sending WAV bytes with `audio/wav` causes an invalid argument
error because the server interprets the WAV header as audio samples.

**Fix:** Record with `sounddevice` using `dtype='int16'`, then call `.tobytes()`
to get raw PCM, and set mime to `audio/pcm;rate=16000`:

```python
audio = sd.rec(..., dtype='int16')
sd.wait()
pcm_bytes = audio.tobytes()   # no WAV header
adapter.transcribe(pcm_bytes, 'audio/pcm;rate=16000')
```

---

### 5. `numpy` not available in the uv run environment

**Error:**
```
ImportError: NumPy must be installed for play()/rec()/playrec()
```

**Cause:** `sounddevice`'s recording functions require `numpy`, which is not
installed in the isolated uv environment by default.

**Fix:** Add `--with numpy` to the `uv run` command alongside `--with sounddevice`.

---

## How the Tests Exercise the Trust-Level Boundary

The test suite enforces the **trust boundary** between the adapter and the real
Gemini Live upstream. The adapter is only trusted to talk to the upstream if it
follows the exact wire contract. The mock acts as the boundary guard.

### The boundary rule

> The Gemini Live API **rejects** any WebSocket session where audio data arrives
> before the `BidiGenerateContentSetup` frame. Order is mandatory.

### How each test enforces the boundary

| Test | What it checks |
|---|---|
| `test_provider_name_matches` | The adapter identifies itself as `"gemini_live"` — callers rely on this to route to the right provider. |
| `test_transcribe_returns_transcribe_result` | The adapter returns a `TranscribeResult` with `provider="gemini_live"` and `language="en"` — contract shape is stable. |
| `test_transcribe_passes_audio_to_upstream` | The adapter actually forwards the audio bytes to the upstream — it cannot silently drop audio. |
| `test_transcribe_records_duration_ms` | The adapter propagates `duration_ms` from the upstream response — callers use this for billing and metrics. |
| `test_transcribe_propagates_upstream_error` | If the upstream returns a status code (e.g. 500), the adapter re-raises it as `STTError(status=500)` — callers must not swallow upstream failures. |
| `test_transcribe_handles_empty_audio` | Empty audio must not crash the adapter — it should return a valid (possibly empty-text) `TranscribeResult`. |
| `test_channel_specific_behaviour_setup_frame_first` | **The key boundary test.** Asserts that `frames_sent[0]` is the setup frame, not an audio frame. The mock's `record_frame()` captures every frame the adapter sends. If audio arrives first, the real server closes the session with 1007. |

### How the mock enforces this

`GeminiLiveMock.record_frame()` appends every frame to `frames_sent` in order.
The adapter calls it in strict order:

```python
mock.record_frame(self._build_setup_frame())   # index 0 — always setup
mock.record_frame(self._build_audio_frame(...)) # index 1 — always audio
```

The behavioural test then asserts `"setup" in frames_sent[0]`, confirming the
adapter respects the ordering requirement without ever hitting the real network.

---

## Setup and Running Tests

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager

### Install dependencies

```powershell
cd glc_v1
uv sync
```

### Run the unit tests (no API key needed)

All 7 tests run against the in-process mock:

```powershell
uv run pytest tests/voice/stt/test_gemini_live.py -v
```

Expected output:
```
tests/voice/stt/test_gemini_live.py::test_provider_name_matches PASSED
tests/voice/stt/test_gemini_live.py::test_transcribe_returns_transcribe_result PASSED
tests/voice/stt/test_gemini_live.py::test_transcribe_passes_audio_to_upstream PASSED
tests/voice/stt/test_gemini_live.py::test_transcribe_records_duration_ms PASSED
tests/voice/stt/test_gemini_live.py::test_transcribe_propagates_upstream_error PASSED
tests/voice/stt/test_gemini_live.py::test_transcribe_handles_empty_audio PASSED
tests/voice/stt/test_gemini_live.py::test_channel_specific_behaviour_setup_frame_first PASSED
7 passed
```

### Live microphone test (requires `GEMINI_API_KEY`)

Records 5 seconds from the default microphone and transcribes it in real time.
The script is the same on all platforms — only how you set the env var differs.

**Windows (PowerShell)**
```powershell
$env:GEMINI_API_KEY = "your-key-here"

uv run --with sounddevice --with numpy python -c "
import asyncio, sounddevice as sd
from glc.voice.stt.providers.gemini_live.adapter import Provider

SAMPLE_RATE = 16000
DURATION    = 5

print('Recording for 5 seconds... speak now!')
audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
sd.wait()
print('Done. Transcribing...')

pcm_bytes = audio.tobytes()
adapter = Provider(config={'ssl_verify': False})
result = asyncio.run(adapter.transcribe(pcm_bytes, 'audio/pcm;rate=16000'))
print('Transcript :', result.text)
print('Duration ms:', result.duration_ms)
"
```

**Linux / macOS (bash / zsh)**
```bash
export GEMINI_API_KEY="your-key-here"

uv run --with sounddevice --with numpy python -c "
import asyncio, sounddevice as sd
from glc.voice.stt.providers.gemini_live.adapter import Provider

SAMPLE_RATE = 16000
DURATION    = 5

print('Recording for 5 seconds... speak now!')
audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
sd.wait()
print('Done. Transcribing...')

pcm_bytes = audio.tobytes()
adapter = Provider(config={})          # ssl_verify not needed on most Linux/Mac
result = asyncio.run(adapter.transcribe(pcm_bytes, 'audio/pcm;rate=16000'))
print('Transcript :', result.text)
print('Duration ms:', result.duration_ms)
"
```

> **Linux note:** `sounddevice` requires PortAudio. Install it first:
> ```bash
> # Debian / Ubuntu
> sudo apt install portaudio19-dev
> # macOS
> brew install portaudio
> ```

> **`ssl_verify: False`** is only needed behind a corporate proxy that injects a
> non-standard CA certificate (e.g. Infoblox network). Remove it on home/cloud networks.

### Config reference

| Key | Type | Default | Description |
|---|---|---|---|
| `api_key` | `str` | `$GEMINI_API_KEY` env var | Gemini API key |
| `model` | `str` | `models/gemini-3.1-flash-live-preview` | Live-capable model string |
| `response_modalities` | `list[str]` | `["AUDIO"]` | Must be `["AUDIO"]` for native audio models |
| `language` | `str` | `"en"` | Language tag returned in `TranscribeResult` |
| `ssl_verify` | `bool` | `True` | Set to `False` to bypass corporate proxy CA errors |
| `mock` | `GeminiLiveMock` | `None` | Inject a mock; skips real WebSocket (used in tests) |
