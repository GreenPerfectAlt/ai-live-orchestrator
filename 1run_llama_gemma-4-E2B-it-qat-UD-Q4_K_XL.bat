@echo off
setlocal EnableExtensions EnableDelayedExpansion

:: ═══════════════════════════════════════════════════════════════
:: ai-live-orchestrator — Gemma-4-E2B QAT launcher (УПРОЩЁННЫЙ)
:: ═══════════════════════════════════════════════════════════════

:: Путь к папке скрипта (для кэшей и относительных ссылок)
for %%I in ("%~dp0") do set "PROJECT_ROOT=%%~sI"
cd /d "%PROJECT_ROOT%" || (
  echo [ERROR] Cannot enter project folder: %PROJECT_ROOT%
  pause
  exit /b 1
)

set "LLM_BACKEND=llama_cpp"
set "LAUNCHER_NAME=run_llama_gemma-4-E2B-it-qat-UD-Q4_K_XL.bat"
set "MODEL_LABEL=Gemma-4-E2B QAT llama.cpp"

:: ─── ПОИСК PYTHON ───────────────────────────────────────────────
:: (оставляем как есть, он рабочий)
set "PORTABLE_PYTHON_EXE="
set "PORTABLE_PYTHON_DIR="

if exist "%PROJECT_ROOT%.venv\Scripts\python.exe" (
  set "PORTABLE_PYTHON_EXE=%PROJECT_ROOT%.venv\Scripts\python.exe"
  set "PORTABLE_PYTHON_DIR=%PROJECT_ROOT%.venv\Scripts"
  echo [PYTHON] Found .venv
) else if exist "C:\AI\Python\Python311\python.exe" (
  set "PORTABLE_PYTHON_EXE=C:\AI\Python\Python311\python.exe"
  set "PORTABLE_PYTHON_DIR=C:\AI\Python\Python311"
  echo [PYTHON] Found portable C:\AI\Python\Python311
) else if exist "%PROJECT_ROOT%..\Python311\python.exe" (
  set "PORTABLE_PYTHON_EXE=%PROJECT_ROOT%..\Python311\python.exe"
  set "PORTABLE_PYTHON_DIR=%PROJECT_ROOT%..\Python311"
  echo [PYTHON] Found relative ..\Python311
) else (
  where python >nul 2>nul
  if errorlevel 1 (
    echo [ERROR] Python not found. Create .venv or place python folder.
    pause
    exit /b 1
  )
  set "PORTABLE_PYTHON_EXE=python"
  set "PORTABLE_PYTHON_DIR=%PROJECT_ROOT%"
  echo [PYTHON] Using system PATH python
)

echo [PYTHON] Executable: %PORTABLE_PYTHON_EXE%

:: ─── ПУТИ К МОДЕЛЯМ (ЖЁСТКИЕ, ПРОВЕРЕННЫЕ) ──────────────────
:: Модель и mmproj лежат в E:\AI\0\@\
set "MODEL_PATH=C:\AI\0\Gemma_4\gemma-4-E2B\gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf"
set "MM_PROJ_PATH=C:\AI\0\Gemma_4\gemma-4-E2B\mmproj-F16-gemma-4-E2B-it-qat.gguf"

:: llama-server.exe лежит в E:\AI\llama\
set "LLAMA_SERVER_EXE=C:\AI\llama\llama-server.exe"

:: ─── ПРОВЕРКА (только чтобы убедиться, что файлы существуют) ──
if not exist "%MODEL_PATH%" (
  echo [ERROR] Модель не найдена: %MODEL_PATH%
  pause
  exit /b 1
)
if not exist "%MM_PROJ_PATH%" (
  echo [ERROR] mmproj не найден: %MM_PROJ_PATH%
  pause
  exit /b 1
)
if not exist "%LLAMA_SERVER_EXE%" (
  echo [ERROR] llama-server.exe не найден: %LLAMA_SERVER_EXE%
  pause
  exit /b 1
)

echo [MODEL]  %MODEL_PATH%
echo [MMPROJ] %MM_PROJ_PATH%
echo [LLAMA]  %LLAMA_SERVER_EXE%

:: ─── НАСТРОЙКИ LLAMA-SERVER (без изменений) ──────────────────
set "LLAMA_MODEL=gemma-4-E2B-it-qat"
set "LLAMA_BASE_URL=http://127.0.0.1:8080/v1"
set "LLAMA_STREAMING=1"
set "TEXT_STREAMING=1"
set "LLM_STREAMING=1"
set "LLAMA_ENABLE_AUDIO=1"
set "LLAMA_SEND_AUDIO_WITH_STT=0"
set "LLAMA_ENABLE_IMAGES=1"
set "LLAMA_MAX_IMAGES=3"
set "LLAMA_HISTORY_TURNS=20"
set "LLAMA_STARTUP_TIMEOUT=240"

set "LLAMA_THREADS=6"
set "LLAMA_CTX_SIZE=4096"
set "LLAMA_BATCH_SIZE=256"
set "LLAMA_UBATCH_SIZE=256"
set "LLAMA_N_GPU_LAYERS=-1"
set "LLAMA_FLAGS=-c %LLAMA_CTX_SIZE% --reasoning off --reasoning-budget 0 -ctk q8_0 -ctv q8_0 --temp 1.0 --top-k 0 --top-p 1.0 --min-p 0.05 --typical 1.00 --mirostat 0 --xtc-probability 0.1 --top-n-sigma 1.1 --swa-full --no-ui --poll 100 --prio 3 -t %LLAMA_THREADS% --threads-batch %LLAMA_THREADS% -n -1 -ngl all -fa on -fit off --parallel 1 --keep 1 --port 8080 -b %LLAMA_BATCH_SIZE% -ub %LLAMA_UBATCH_SIZE% --jinja --no-mmproj-offload"


:: ─── TTS НАСТРОЙКИ ──────────────────────────────────────────────
set "TTS_STREAMING=1"
set "TTS_EARLY_CHARS=30"
set "TTS_LONG_CHARS=75"
set "TTS_MAX_CHARS=200"
set "TTS_SPLIT_ON_COMMA=0"
set "TTS_SENTENCE_STREAMING=1"
set "TTS_ENGINE=silero"
set "TTS_LANG=auto"
set "TTS_VOICE=F4"
set "TTS_SPEED=0.98"
set "TTS_STEPS=1"
set "TTS_THREADS=1"
set "TTS_DO_NOT_BLOCK_MODEL=1"
set "TTS_BACKGROUND_PRELOAD=1"

:: Silero
set "SILERO_MODEL=v5_5_ru"
set "SILERO_SPEAKER=xenia"
set "SILERO_SAMPLE_RATE=24000"
set "SILERO_SPEED=1.0"
set "SILERO_PUT_ACCENT=1"
set "SILERO_PUT_YO=1"
set "SILERO_CACHE_DIR=%PROJECT_ROOT%models\silero"
set "SILERO_USE_HUB=1"

:: ─── STT НАСТРОЙКИ ──────────────────────────────────────────────
set "STT_ENGINE=faster_whisper"
set "STT_MODEL=turbo"
set "STT_LANG=ru"
set "STT_COMPUTE_TYPE=int8"
set "STT_BEAM_SIZE=3"
set "STT_VAD_ENABLE=1"
set "STT_VAD_SILENCE_MS=450"
set "STT_DEVICE=cpu"
set "STT_THREADS=1"

:: ─── ОПТИМИЗАЦИЯ CPU / OMP ──────────────────────────────────────
set "PYTHONOPTIMIZE=2"
set "ORT_DISABLE_TELEMETRY=1"
set "OMP_WAIT_POLICY=ACTIVE"
set "OMP_PROC_BIND=CLOSE"
set "OMP_PLACES=cores"
set "KMP_BLOCKTIME=0"
set "OMP_NUM_THREADS=4"
set "MKL_NUM_THREADS=4"
set "IN_OUT_THREADS=2"

:: ─── КЭШИ / ЛОКАЛИЗАЦИЯ ─────────────────────────────────────────
set "HF_HOME=%PROJECT_ROOT%models\.hf_cache"
set "HUGGINGFACE_HUB_CACHE=%HF_HOME%\hub"
set "SUPERTONIC_CACHE_DIR=%PROJECT_ROOT%models\supertonic3"
set "TORCH_HOME=%PROJECT_ROOT%models\.torch"
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

if not exist "%HF_HOME%" mkdir "%HF_HOME%"

:: ─── УСТАНОВКА ЗАВИСИМОСТЕЙ (оставляем как есть) ──────────────
"%PORTABLE_PYTHON_EXE%" -c "import torch" >nul 2>nul
if errorlevel 1 (
  echo [SETUP] Installing PyTorch CPU...
  "%PORTABLE_PYTHON_EXE%" -m pip install --index-url https://download.pytorch.org/whl/cpu torch --no-warn-script-location
)

"%PORTABLE_PYTHON_EXE%" -c "import fastapi,uvicorn,numpy,supertonic,websockets,faster_whisper" >nul 2>nul
if errorlevel 1 (
  echo [SETUP] Installing dependencies from requirements.txt ...
  "%PORTABLE_PYTHON_EXE%" -m pip install -r "%PROJECT_ROOT%requirements.txt" --no-warn-script-location
  if errorlevel 1 (
    echo [WARN] requirements.txt failed. Installing core deps...
    "%PORTABLE_PYTHON_EXE%" -m pip install fastapi "uvicorn[standard]" websockets "numpy>=2.0.0" requests "supertonic>=1.3.1" --no-warn-script-location
  )
)

"%PORTABLE_PYTHON_EXE%" -c "import fastapi,uvicorn,websockets" >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Critical Python packages missing.
  pause
  exit /b 1
)

:: ─── ЗАПУСК ─────────────────────────────────────────────────────
echo.
echo [LLAMA] Restarting llama-server...
taskkill /IM llama-server.exe /F >nul 2>nul
timeout /t 1 /nobreak >nul

echo [START] Starting llama-server on 127.0.0.1:8080 ...
start "llama-server E2B" "%LLAMA_SERVER_EXE%" -m "%MODEL_PATH%" --mmproj "%MM_PROJ_PATH%" --alias "%LLAMA_MODEL%" --host 127.0.0.1 %LLAMA_FLAGS%
timeout /t 5 /nobreak >nul

echo ===================================================
echo  ai-live-orchestrator — Gemma-4-E2B QAT
echo  Project:  %PROJECT_ROOT%
echo  Model:    %MODEL_PATH%
echo  Python:   %PORTABLE_PYTHON_EXE%
echo  URL:      http://127.0.0.1:8000
echo ===================================================
echo.

set "FORWARD_ROOT=%PROJECT_ROOT:\=/%"
"%PORTABLE_PYTHON_EXE%" -c "import sys, runpy; sys.path.insert(0, '%FORWARD_ROOT%'); runpy.run_path('%FORWARD_ROOT%server.py', run_name='__main__')"

pause
exit /b 0
exit /b 0