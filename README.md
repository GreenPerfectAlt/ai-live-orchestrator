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
- Gemma 4 GGUF
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

Recommended: Python 3.11 or 3.12.

Python 3.13 is not recommended yet because some audio/TTS/native dependencies may not have stable wheels for it.

### 2. Install dependencies

Recommended:

```bat
install_menu.bat
```

Use the menu:

```text
[1] Install base app dependencies
[2] Install PyTorch CPU for Silero RU
[3] Full install: base deps + PyTorch CPU
[4] Check environment
[5] Reset .venv
[6] Create folders
[0] Exit
```

Fast path:

```bat
install_requirements.bat
```
### 3. Add llama.cpp server

Put llama-server.exe into the project folder.

Expected:

ai-live-orchestrator\llama-server.exe

Or edit LLAMA_SERVER_EXE in run_llama_CUSTOM_TEMPLATE.bat.

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

Edit:

run_llama_CUSTOM_TEMPLATE.bat

Configure the block:

USER CONFIG - REQUIRED

Then run:

run_llama_CUSTOM_TEMPLATE.bat

Open:

http://127.0.0.1:8000

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
## Gemma 4 / GGUF / LiteRT-LM notes

The main public runtime path in this repository is:

llama.cpp
llama-server.exe
*.gguf
mmproj*.gguf
run_llama_CUSTOM_TEMPLATE.bat

Gemma 4 is the target model family for local experiments.

GGUF is used as the practical local model format for llama.cpp.

LiteRT-LM / .litertlm is a related on-device model runtime direction, but the main user-facing setup in this repository is currently based on llama.cpp / GGUF.

Model references:

https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-unquantized

https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-unquantized

## Custom launcher template

This repository includes a user-editable launcher template:

```bat
run_llama_CUSTOM_TEMPLATE.bat
```

The launcher is meant for custom local GGUF setups.

Edit the top block first:

```text
USER CONFIG - REQUIRED
```

Main fields:

```bat
set "LLAMA_SERVER_EXE=%PROJECT_ROOT%llama-server.exe"
set "MODEL_PATH=%PROJECT_ROOT%models\PUT-YOUR-MODEL.gguf"
set "USE_MMPROJ=1"
set "MM_PROJ_PATH=%PROJECT_ROOT%models\PUT-YOUR-MMPROJ.gguf"
set "MODEL_LABEL=PUT-YOUR-MODEL-LABEL"
set "LLAMA_MODEL=PUT-YOUR-MODEL-ALIAS"
```

Meaning:

* `LLAMA_SERVER_EXE` — path to `llama-server.exe`
* `MODEL_PATH` — path to main `*.gguf` model
* `USE_MMPROJ` — `1` to use vision/multimodal projector, `0` to disable it
* `MM_PROJ_PATH` — path to `mmproj*.gguf`
* `MODEL_LABEL` — display name for logs/UI
* `LLAMA_MODEL` — model alias for llama.cpp OpenAI-compatible endpoint

Text-only example:

```bat
set "LLAMA_SERVER_EXE=%PROJECT_ROOT%llama-server.exe"
set "MODEL_PATH=%PROJECT_ROOT%models\my-model.gguf"
set "USE_MMPROJ=0"
set "MM_PROJ_PATH="
set "MODEL_LABEL=My Local Model"
set "LLAMA_MODEL=my-local-model"
```

Multimodal example:

```bat
set "LLAMA_SERVER_EXE=%PROJECT_ROOT%llama-server.exe"
set "MODEL_PATH=%PROJECT_ROOT%models\gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf"
set "USE_MMPROJ=1"
set "MM_PROJ_PATH=%PROJECT_ROOT%models\mmproj-F16-gemma-4-E2B-it-qat.gguf"
set "MODEL_LABEL=Gemma 4 E2B GGUF"
set "LLAMA_MODEL=gemma-4-e2b"
```

The template checks placeholders and required files before launch.

Typical errors:

```text
[ERROR] Replace PUT-YOUR placeholders first
[ERROR] GGUF model not found
[ERROR] mmproj file not found
[ERROR] llama-server.exe not found
```

## Local-first design

AI Live Orchestrator is built around a local-first workflow:

```text
browser UI
WebSocket
FastAPI backend
llama.cpp server
local GGUF model
local TTS
local visual/audio context
```

The core pipeline does not require a cloud AI inference API.

Local files stay on the user's machine unless the user explicitly changes the setup.

## Main changes from upstream Parlor

This project was originally based on the open-source Parlor project by fikrikarim:

```text
https://github.com/fikrikarim/parlor
```

AI Live Orchestrator contains major modifications focused on:

* Windows-first launch flow
* `llama.cpp` / GGUF runtime
* local `llama-server.exe` workflow
* custom `.bat` launcher template
* explicit local model paths
* Gemma 4 GGUF setup
* `mmproj*.gguf` multimodal projector setup
* Silero RU / Supertonic 3 TTS options
* local `models\` folder workflow
* manual large-model handling
* updated voice/multimodal orchestration
* interruption / barge-in behavior
* push-to-talk / mic control behavior

## Acknowledgments

Based on / derived from:

* Parlor by fikrikarim: https://github.com/fikrikarim/parlor

Original Parlor project:

* License: Apache License 2.0
* Runtime idea: on-device real-time multimodal AI
* Original stack direction: browser mic/camera + FastAPI + local model + TTS

Additional technologies used or targeted in this fork:

* Gemma 4 by Google / Google DeepMind
* llama.cpp
* GGUF model format
* Silero RU
* Supertonic 3
* Chrome SpeechRecognition
* FastAPI
* WebSocket

## License

This project is licensed under Apache License 2.0.

See:

 ```text 
 LICENSE
```

The original Parlor project is also licensed under Apache License 2.0.

This repository keeps attribution and documents major modifications.
