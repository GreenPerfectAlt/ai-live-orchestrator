@echo off
setlocal EnableExtensions

set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%" || (
  echo [ERROR] Cannot enter project folder: %PROJECT_ROOT%
  pause
  exit /b 1
)

set "LLM_BACKEND=llama_cpp"
set "LAUNCHER_NAME=run_llama_e2b.bat"
set "MODEL_LABEL=Gemma-4-E2B QAT llama.cpp"
set "MODEL_PATH=%PROJECT_ROOT%models\gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf"
set "MM_PROJ_PATH=%PROJECT_ROOT%models\mmproj-F16-gemma-4-E2B-it-qat"
if not exist "%MM_PROJ_PATH%" if exist "%PROJECT_ROOT%models\mmproj-F16-gemma-4-E2B-it-qat.gguf" set "MM_PROJ_PATH=%PROJECT_ROOT%models\mmproj-F16-gemma-4-E2B-it-qat.gguf"

set "LLAMA_SERVER_EXE=%PROJECT_ROOT%llama-server.exe"
set "LLAMA_MODEL=gemma-4-E2B-it-qat"
set "LLAMA_BASE_URL=http://127.0.0.1:8080/v1"
set "LLAMA_STREAMING=1"
set "TEXT_STREAMING=1"
set "LLM_STREAMING=1"
set "LLAMA_ENABLE_AUDIO=1"
set "LLAMA_SEND_AUDIO_WITH_STT=0"
set "TTS_STREAMING=1"
set "TTS_EARLY_CHARS=80"
set "TTS_LONG_CHARS=180"
set "LLAMA_ENABLE_IMAGES=1"
set "LLAMA_MAX_IMAGES=6"
set "LLAMA_HISTORY_TURNS=32"
set "LLAMA_STARTUP_TIMEOUT=240"

rem Ryzen 5 5500U: 6 threads is usually the sweet spot. Change if needed.
set "LLAMA_THREADS=6"
set "LLAMA_CTX_SIZE=8096"
set "LLAMA_BATCH_SIZE=512"
set "LLAMA_UBATCH_SIZE=1024"
set "LLAMA_N_GPU_LAYERS=-1"
set "LLAMA_FLAGS=-c 8096 --reasoning off --reasoning-budget 0 -ctk q8_0 -ctv q8_0 --temp 1.0 --top-k 0 --top-p 1.0 --min-p 0.08 --typical 1.00 --dynatemp-range 0.00 --mirostat 0 --spec-type ngram-simple --spec-draft-n-max 3 --ctx-checkpoints 0 --no-warmup -n -1 -ngl -1 --prio 3 -fa on -fit on -t 6 --threads-batch 6 -cmoe --parallel 1 --keep 1 --port 8080 -b 512 -ub 1024 --jinja --no-mmproj-offload"

set "HF_HOME=%PROJECT_ROOT%models\.hf_cache"
set "HUGGINGFACE_HUB_CACHE=%HF_HOME%\hub"
set "SUPERTONIC_CACHE_DIR=%PROJECT_ROOT%models\supertonic3"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
set "HF_HUB_DISABLE_XET=1"
set "HF_HUB_ENABLE_HF_TRANSFER=0"
set "HF_HUB_DOWNLOAD_TIMEOUT=1200"
set "HF_HUB_ETAG_TIMEOUT=60"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "GLOG_minloglevel=2"
set "TF_CPP_MIN_LOG_LEVEL=2"
set "ABSL_MIN_LOG_LEVEL=2"

set "TTS_MAX_CHARS=260"
set "TTS_SPLIT_ON_COMMA=0"
set "TTS_ENGINE=supertonic"
set "SILERO_MODEL=v5_5_ru"
set "SILERO_MODEL_URL=https://models.silero.ai/models/tts/ru/v5_5_ru.pt"
set "SILERO_SPEAKER=baya"
set "SILERO_SAMPLE_RATE=24000"
set "SILERO_SPEED=1.00"
set "SILERO_PUT_ACCENT=1"
set "SILERO_PUT_YO=1"
set "SILERO_CACHE_DIR=%PROJECT_ROOT%models\silero"
set "TORCH_HOME=%PROJECT_ROOT%models\.torch"
set "TTS_THREADS=3"
set "SILERO_USE_HUB=1"
set "TTS_DO_NOT_BLOCK_MODEL=1"
set "TTS_BACKGROUND_PRELOAD=1"
set "TTS_SENTENCE_STREAMING=1"
set "TTS_LANG=auto"
set "TTS_VOICE=F4"
set "TTS_SPEED=0.96"
set "TTS_STEPS=6"

set "LLM_TEMPERATURE=1"
set "LLM_TOP_P=1.0"
set "LLM_TOP_K=0"
set "LLM_MIN_P=0.08"
set "LLM_TYPICAL_P=1.00"
set "LLM_MAX_OUTPUT_TOKENS=0"
set "LLM_ENABLE_THINKING=0"

if not exist "%PROJECT_ROOT%models" mkdir "%PROJECT_ROOT%models"
if not exist "%HF_HOME%" mkdir "%HF_HOME%"

if not exist "%MODEL_PATH%" (
  echo [ERROR] GGUF model not found:
  echo %MODEL_PATH%
  echo.
  echo Put gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf into the models folder.
  pause
  exit /b 1
)

if not exist "%MM_PROJ_PATH%" (
  echo [ERROR] mmproj file not found:
  echo %MM_PROJ_PATH%
  echo.
  echo Put mmproj-F16-gemma-4-E2B-it-qat into the models folder.
  pause
  exit /b 1
)

if not defined LLAMA_SERVER_EXE (
  if exist "%PROJECT_ROOT%llama-server.exe" set "LLAMA_SERVER_EXE=%PROJECT_ROOT%llama-server.exe"
  if not defined LLAMA_SERVER_EXE if exist "%PROJECT_ROOT%llama.cpp\build\bin\Release\llama-server.exe" set "LLAMA_SERVER_EXE=%PROJECT_ROOT%llama-server.exe"
  if not defined LLAMA_SERVER_EXE if exist "%PROJECT_ROOT%llama.cpp\build\bin\llama-server.exe" set "LLAMA_SERVER_EXE=%PROJECT_ROOT%llama-server.exe"
  if not defined LLAMA_SERVER_EXE set "LLAMA_SERVER_EXE=%PROJECT_ROOT%llama-server.exe"
)

if exist "%LLAMA_SERVER_EXE%" goto llama_exe_ok
where llama-server.exe >nul 2>nul
if not errorlevel 1 goto llama_exe_ok

echo [ERROR] llama-server.exe not found.
echo Put llama-server.exe near this .bat, or set LLAMA_SERVER_EXE to the full path.
pause
exit /b 1

:llama_exe_ok
if not exist "%PROJECT_ROOT%.venv\Scripts\python.exe" (
  call :create_venv
  if errorlevel 1 (
    pause
    exit /b 1
  )
)

call "%PROJECT_ROOT%.venv\Scripts\activate.bat" || (
  echo [ERROR] Cannot activate .venv
  pause
  exit /b 1
)

python -c "import fastapi,uvicorn,numpy" >nul 2>nul
if errorlevel 1 (
  echo [SETUP] Installing backend dependencies into .venv...
  python -m pip install -U pip setuptools wheel
  if errorlevel 1 goto pip_failed
  python -m pip install fastapi "uvicorn[standard]" "numpy>=2.0.0"
  if errorlevel 1 goto pip_failed
)

python -c "import supertonic" >nul 2>nul
if errorlevel 1 (
  echo [SETUP] Installing Supertonic optional backend...
  python -m pip install "supertonic>=1.3.1"
  if errorlevel 1 echo [WARN] Supertonic install failed. Silero can still work.
)

python -c "import torch" >nul 2>nul
if errorlevel 1 (
  echo [SETUP] Installing PyTorch CPU for Silero TTS. This is large and happens once...
  python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
  if errorlevel 1 (
    echo [SETUP] PyTorch CPU index failed, trying normal PyPI torch...
    python -m pip install torch
    if errorlevel 1 goto pip_failed
  )
)


echo [LLAMA] Restarting llama-server so the selected bat/model/settings are really applied...
taskkill /IM llama-server.exe /F >nul 2>nul
timeout /t 1 /nobreak >nul
echo [CHECK] Looking for already running llama-server...
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { Invoke-RestMethod '%LLAMA_BASE_URL%/models' -TimeoutSec 2 ^| Out-Null; exit 0 } catch { exit 1 }" >nul 2>nul
if errorlevel 1 (
  echo [START] Starting llama-server E2B on 127.0.0.1:8080 ...
  start "llama-server E2B" cmd /k ""%LLAMA_SERVER_EXE%" -m "%MODEL_PATH%" --mmproj "%MM_PROJ_PATH%" --alias "%LLAMA_MODEL%" --host 127.0.0.1 %LLAMA_FLAGS%"
  timeout /t 3 /nobreak >nul
) else (
  echo [OK] llama-server already running.
)

echo ===================================================
echo [Parlor llama.cpp] Starting Gemma-4-E2B QAT
echo Project:      %PROJECT_ROOT%
echo Model:        %MODEL_PATH%
echo mmproj:       %MM_PROJ_PATH%
echo llama API:    %LLAMA_BASE_URL%
echo Threads:      %LLAMA_THREADS%
echo Parlor URL:   http://127.0.0.1:8000
echo Streaming:    ON ^(text_delta + early sentence/phrase TTS^)
echo ===================================================

python server.py
pause
exit /b 0

:create_venv
echo [SETUP] Virtual environment not found. Creating .venv...
where py >nul 2>nul
if not errorlevel 1 (
  py -3.12 -m venv "%PROJECT_ROOT%.venv" >nul 2>nul
  if exist "%PROJECT_ROOT%.venv\Scripts\python.exe" exit /b 0
  py -3.11 -m venv "%PROJECT_ROOT%.venv" >nul 2>nul
  if exist "%PROJECT_ROOT%.venv\Scripts\python.exe" exit /b 0
)
where python >nul 2>nul
if not errorlevel 1 (
  python -m venv "%PROJECT_ROOT%.venv"
  if exist "%PROJECT_ROOT%.venv\Scripts\python.exe" exit /b 0
)
echo [ERROR] Python 3.11 or 3.12 was not found, or venv creation failed.
echo Install Python 3.11/3.12 and enable "Add python.exe to PATH".
exit /b 1

:pip_failed
echo.
echo [ERROR] Dependency installation failed.
echo Check internet connection and Python version. Use Python 3.11 or 3.12.
exit /b 1
