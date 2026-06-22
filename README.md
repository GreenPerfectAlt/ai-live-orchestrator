# AI Live Orchestrator

Local multimodal voice AI app for Windows.

Runs a local Gemma/GGUF model through `llama.cpp` and provides:

- live chat UI
- WebSocket streaming
- local voice input
- STT text / Native audio / Hybrid audio modes
- screen / camera / image / PDF / video context
- local TTS through Silero RU / Supertonic 3
- interruption / barge-in logic
- Windows `.bat` launch scripts

## Stack

- Python
- FastAPI
- WebSocket
- JavaScript
- HTML/CSS
- llama.cpp
- Gemma GGUF
- mmproj
- Silero RU
- Supertonic 3
- Chrome SpeechRecognition
- Windows Batch

## What is not included

Large model files are not included.

Do not commit:

```gitignore
models/*.gguf
models/*.pt
models/*.bin
models/*.safetensors
models/.hf_cache/
models/.torch/
models/supertonic3/
.venv/
```

## Setup

### 1. Install Python

Use Python `3.11` or `3.12`.

### 2. Install dependencies

```bat
install_requirements.bat
```

### 3. Add llama.cpp server

Put `llama-server.exe` into the project folder.

Expected:

```text
ai-live-orchestrator\llama-server.exe
```

Or edit `run_llama*.bat`.

### 4. Add models

Put model files into:

```text
ai-live-orchestrator\models\
```

You need:

```text
*.gguf
mmproj*.gguf
```

See:

```text
models\PUT_MODELS_HERE.txt
```

### 5. Run

Example:

```bat
run_llama_e2b.bat
```

Open:

```text
http://127.0.0.1:8000
```

## Voice modes

`Gemma mic input mode`:

- `STT text` — sends Chrome speech-to-text result only
- `Native audio` — sends WAV directly to Gemma through `input_audio`
- `Hybrid` — sends STT text + WAV together

## Silero

Default newest model env:

```bat
set "SILERO_MODEL=v5_5_ru"
set "SILERO_MODEL_URL=https://models.silero.ai/models/tts/ru/v5_5_ru.pt"
```

## License

Add your license here.
