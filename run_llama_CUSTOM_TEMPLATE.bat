@echo off
setlocal EnableExtensions

rem ===================================================
rem AI Live Orchestrator - custom llama.cpp launcher
rem Edit ONLY the USER CONFIG section first.
rem Keep this file as ASCII/UTF-8 without Cyrillic comments.
rem Put this .bat in the project root near server.py and index.html.
rem ===================================================

set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%" || (
  echo [ERROR] Cannot enter project folder: %PROJECT_ROOT%
  pause
  exit /b 1
)

rem ===================================================
rem USER CONFIG - REQUIRED
rem ===================================================

rem 1) Path to llama-server.exe.
rem Default expects llama-server.exe in the project root.
rem Example: set "LLAMA_SERVER_EXE=C:\AI\llama.cpp\build\bin\Release\llama-server.exe"
set "LLAMA_SERVER_EXE=%PROJECT_ROOT%llama-server.exe"

rem 2) Path to main GGUF model.
rem Example E2B:  set "MODEL_PATH=%PROJECT_ROOT%models\gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf"
rem Example E4B:  set "MODEL_PATH=%PROJECT_ROOT%models\gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf"
rem Example 12B:  set "MODEL_PATH=%PROJECT_ROOT%models\gemma-4-12B-it-qat-UD-Q4_K_XL.gguf"
set "MODEL_PATH=%PROJECT_ROOT%models\PUT-YOUR-MODEL.gguf"

rem 3) Path to multimodal projector.
rem Required for camera/screen/image/PDF/video context.
rem If your model is text-only, set USE_MMPROJ=0.
rem Example E2B:  set "MM_PROJ_PATH=%PROJECT_ROOT%models\mmproj-F16-gemma-4-E2B-it-qat.gguf"
rem Example E4B:  set "MM_PROJ_PATH=%PROJECT_ROOT%models\mmproj-F16-gemma-4-E4B-it-qat.gguf"
rem Example 12B:  set "MM_PROJ_PATH=%PROJECT_ROOT%models\mmproj-F16-gemma-4-12B-it-qat.gguf"
set "USE_MMPROJ=1"
set "MM_PROJ_PATH=%PROJECT_ROOT%models\PUT-YOUR-MMPROJ.gguf"

rem 4) Human-readable label shown in logs/UI.
rem Example: set "MODEL_LABEL=Gemma-4-E2B QAT llama.cpp custom"
set "MODEL_LABEL=PUT-YOUR-MODEL-LABEL"

rem 5) llama.cpp model alias used by /v1/chat/completions.
rem Example: set "LLAMA_MODEL=gemma-4-E2B-it-qat"
set "LLAMA_MODEL=PUT-YOUR-MODEL-ALIAS"

rem ===================================================
rem USER CONFIG - PERFORMANCE
rem ===================================================

rem Port used by llama-server. Keep 8080 unless you know why.
set "LLAMA_HOST=127.0.0.1"
set "LLAMA_PORT=8080"
set "LLAMA_BASE_URL=http://%LLAMA_HOST%:%LLAMA_PORT%/v1"

rem Threads: Ryzen 5 5500U usually likes 6.
set "LLAMA_THREADS=6"
set "LLAMA_THREADS_BATCH=6"

rem Context size. Lower if RAM is tight.
rem Safe: 4096 / 8192. Heavy: 12000.
set "LLAMA_CTX_SIZE=8192"

rem Batch / microbatch. Lower if launch crashes or swaps.
set "LLAMA_BATCH_SIZE=512"
set "LLAMA_UBATCH_SIZE=512"

rem GPU layers. -1 = offload as much as possible. 0 = CPU only.
set "LLAMA_N_GPU_LAYERS=-1"

rem Flash attention. Use on for speed, off if llama.cpp build rejects it.
set "LLAMA_FLASH_ATTN=on"

rem Keep 0 unless you know ctx checkpoints help your build.
set "LLAMA_CTX_CHECKPOINTS=0"

rem Kill old llama-server before starting this config.
rem 1 = always apply selected model/settings. 0 = reuse existing server if reachable.
set "AUTO_KILL_LLAMA=1"

rem ===================================================
rem USER CONFIG - SAMPLING
rem ===================================================

set "LLM_TEMPERATURE=1.0"
set "LLM_TOP_P=1.0"
set "LLM_TOP_K=0"
set "LLM_MIN_P=0.05"
set "LLM_TYPICAL_P=1.00"
set "LLM_MAX_OUTPUT_TOKENS=0"
set "LLM_ENABLE_THINKING=0"

rem ===================================================
rem USER CONFIG - VOICE / TTS
rem ===================================================

rem TTS_ENGINE options: supertonic / silero
set "TTS_ENGINE=supertonic"

rem Supertonic settings. F1-F5 female, M1-M5 male. auto keeps English terms in Latin better.
set "TTS_LANG=auto"
set "TTS_VOICE=F4"
set "TTS_SPEED=0.96"
set "TTS_STEPS=6"

rem Silero fallback settings. Speakers: baya, xenia, kseniya, aidar, eugene.
set "SILERO_MODEL=v5_5_ru"
set "SILERO_MODEL_URL=https://models.silero.ai/models/tts/ru/v5_5_ru.pt"
set "SILERO_SPEAKER=xenia"
set "SILERO_SAMPLE_RATE=24000"
set "SILERO_SPEED=0.94"
set "SILERO_PUT_ACCENT=1"
set "SILERO_PUT_YO=1"

rem TTS streaming. Lower TTS_EARLY_CHARS = faster first voice, more risk of choppy speech.
set "TTS_STREAMING=1"
set "TTS_SENTENCE_STREAMING=1"
set "TTS_DO_NOT_BLOCK_MODEL=1"
set "TTS_BACKGROUND_PRELOAD=1"
set "TTS_EARLY_CHARS=80"
set "TTS_LONG_CHARS=180"
set "TTS_MAX_CHARS=220"
set "TTS_SPLIT_ON_COMMA=0"
set "TTS_THREADS=2"

rem ===================================================
rem ADVANCED APP SETTINGS
rem ===================================================

set "LLM_BACKEND=llama_cpp"
set "LAUNCHER_NAME=%~nx0"
set "LLAMA_STREAMING=1"
set "TEXT_STREAMING=1"
set "LLM_STREAMING=1"
set "LLAMA_ENABLE_AUDIO=1"
set "LLAMA_SEND_AUDIO_WITH_STT=0"
set "LLAMA_ENABLE_IMAGES=1"
set "LLAMA_MAX_IMAGES=6"
set "LLAMA_HISTORY_TURNS=32"
set "LLAMA_STARTUP_TIMEOUT=240"
set "LLAMA_REQUEST_TIMEOUT=600"

rem Local caches. Kept inside project folder.
set "HF_HOME=%PROJECT_ROOT%models\.hf_cache"
set "HUGGINGFACE_HUB_CACHE=%HF_HOME%\hub"
set "SUPERTONIC_CACHE_DIR=%PROJECT_ROOT%models\supertonic3"
set "SILERO_CACHE_DIR=%PROJECT_ROOT%models\silero"
set "TORCH_HOME=%PROJECT_ROOT%models\.torch"

rem Windows/Python/HF safety.
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

rem Build llama.cpp flags from variables above.
set "LLAMA_FLAGS=-c %LLAMA_CTX_SIZE%"
set "LLAMA_FLAGS=%LLAMA_FLAGS% --reasoning off --reasoning-budget 0"
set "LLAMA_FLAGS=%LLAMA_FLAGS% -ctk q8_0 -ctv q8_0"
set "LLAMA_FLAGS=%LLAMA_FLAGS% --temp %LLM_TEMPERATURE% --top-k %LLM_TOP_K% --top-p %LLM_TOP_P% --min-p %LLM_MIN_P% --typical %LLM_TYPICAL_P%"
set "LLAMA_FLAGS=%LLAMA_FLAGS% --dynatemp-range 0.00 --mirostat 0"
set "LLAMA_FLAGS=%LLAMA_FLAGS% --spec-type ngram-simple --spec-draft-n-max 3"
set "LLAMA_FLAGS=%LLAMA_FLAGS% --ctx-checkpoints %LLAMA_CTX_CHECKPOINTS%"
set "LLAMA_FLAGS=%LLAMA_FLAGS% -n -1 -ngl %LLAMA_N_GPU_LAYERS% --prio 3"
set "LLAMA_FLAGS=%LLAMA_FLAGS% -fa %LLAMA_FLASH_ATTN% -fit on"
set "LLAMA_FLAGS=%LLAMA_FLAGS% -t %LLAMA_THREADS% --threads-batch %LLAMA_THREADS_BATCH%"
set "LLAMA_FLAGS=%LLAMA_FLAGS% --parallel 1 --keep 1 --port %LLAMA_PORT%"
set "LLAMA_FLAGS=%LLAMA_FLAGS% -b %LLAMA_BATCH_SIZE% -ub %LLAMA_UBATCH_SIZE%"
set "LLAMA_FLAGS=%LLAMA_FLAGS% --jinja --no-mmproj-offload"

rem ===================================================
rem VALIDATION
rem ===================================================

if not exist "%PROJECT_ROOT%models" mkdir "%PROJECT_ROOT%models"
if not exist "%HF_HOME%" mkdir "%HF_HOME%"
if not exist "%SILERO_CACHE_DIR%" mkdir "%SILERO_CACHE_DIR%"
if not exist "%SUPERTONIC_CACHE_DIR%" mkdir "%SUPERTONIC_CACHE_DIR%"

if /I "%MODEL_PATH%"=="%PROJECT_ROOT%models\PUT-YOUR-MODEL.gguf" (
  echo [ERROR] Edit MODEL_PATH in USER CONFIG.
  echo Example: set "MODEL_PATH=%%PROJECT_ROOT%%models\gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf"
  pause
  exit /b 1
)

if /I "%USE_MMPROJ%"=="1" if /I "%MM_PROJ_PATH%"=="%PROJECT_ROOT%models\PUT-YOUR-MMPROJ.gguf" (
  echo [ERROR] Edit MM_PROJ_PATH in USER CONFIG, or set USE_MMPROJ=0 for text-only mode.
  pause
  exit /b 1
)

if /I "%MODEL_LABEL%"=="PUT-YOUR-MODEL-LABEL" (
  echo [ERROR] Edit MODEL_LABEL in USER CONFIG.
  pause
  exit /b 1
)

if /I "%LLAMA_MODEL%"=="PUT-YOUR-MODEL-ALIAS" (
  echo [ERROR] Edit LLAMA_MODEL in USER CONFIG.
  pause
  exit /b 1
)

if not exist "%MODEL_PATH%" (
  echo [ERROR] GGUF model not found:
  echo %MODEL_PATH%
  echo.
  echo Fix MODEL_PATH or put the .gguf file there.
  pause
  exit /b 1
)

if /I "%USE_MMPROJ%"=="1" if not exist "%MM_PROJ_PATH%" (
  echo [ERROR] mmproj file not found:
  echo %MM_PROJ_PATH%
  echo.
  echo Fix MM_PROJ_PATH, put the mmproj file there, or set USE_MMPROJ=0.
  pause
  exit /b 1
)

if not exist "%LLAMA_SERVER_EXE%" (
  if exist "%PROJECT_ROOT%llama-server.exe" set "LLAMA_SERVER_EXE=%PROJECT_ROOT%llama-server.exe"
)
if not exist "%LLAMA_SERVER_EXE%" (
  if exist "%PROJECT_ROOT%llama.cpp\build\bin\Release\llama-server.exe" set "LLAMA_SERVER_EXE=%PROJECT_ROOT%llama.cpp\build\bin\Release\llama-server.exe"
)
if not exist "%LLAMA_SERVER_EXE%" (
  if exist "%PROJECT_ROOT%llama.cpp\build\bin\llama-server.exe" set "LLAMA_SERVER_EXE=%PROJECT_ROOT%llama.cpp\build\bin\llama-server.exe"
)
if not exist "%LLAMA_SERVER_EXE%" (
  where llama-server.exe >nul 2>nul
  if errorlevel 1 (
    echo [ERROR] llama-server.exe not found.
    echo Put llama-server.exe near this .bat, or edit LLAMA_SERVER_EXE in USER CONFIG.
    pause
    exit /b 1
  ) else (
    set "LLAMA_SERVER_EXE=llama-server.exe"
  )
)

rem ===================================================
rem PYTHON VENV / DEPENDENCIES
rem ===================================================

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
  python -m pip install fastapi "uvicorn[standard]" "numpy>=2.0.0" requests
  if errorlevel 1 goto pip_failed
)

python -c "import supertonic" >nul 2>nul
if errorlevel 1 (
  echo [SETUP] Installing Supertonic optional backend...
  python -m pip install "supertonic>=1.3.1"
  if errorlevel 1 echo [WARN] Supertonic install failed. Silero can still work.
)

if /I "%TTS_ENGINE%"=="silero" (
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
)

rem ===================================================
rem START LLAMA-SERVER
rem ===================================================

if "%AUTO_KILL_LLAMA%"=="1" (
  echo [LLAMA] Killing old llama-server.exe so selected settings apply...
  taskkill /IM llama-server.exe /F >nul 2>nul
  timeout /t 1 /nobreak >nul
)

echo [CHECK] Looking for already running llama-server at %LLAMA_BASE_URL% ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { Invoke-RestMethod '%LLAMA_BASE_URL%/models' -TimeoutSec 2 ^| Out-Null; exit 0 } catch { exit 1 }" >nul 2>nul
if errorlevel 1 (
  echo [START] Starting llama-server: %MODEL_LABEL%
  if /I "%USE_MMPROJ%"=="1" (
    start "llama-server %LLAMA_MODEL%" cmd /k ""%LLAMA_SERVER_EXE%" -m "%MODEL_PATH%" --mmproj "%MM_PROJ_PATH%" --alias "%LLAMA_MODEL%" --host %LLAMA_HOST% %LLAMA_FLAGS%"
  ) else (
    start "llama-server %LLAMA_MODEL%" cmd /k ""%LLAMA_SERVER_EXE%" -m "%MODEL_PATH%" --alias "%LLAMA_MODEL%" --host %LLAMA_HOST% %LLAMA_FLAGS%"
  )
  timeout /t 3 /nobreak >nul
) else (
  echo [OK] llama-server already running.
)

rem ===================================================
rem START AI LIVE ORCHESTRATOR
rem ===================================================

echo ===================================================
echo [AI Live Orchestrator] Starting %MODEL_LABEL%
echo Project:      %PROJECT_ROOT%
echo Model:        %MODEL_PATH%
echo mmproj:       %MM_PROJ_PATH%
echo Use mmproj:   %USE_MMPROJ%
echo llama API:    %LLAMA_BASE_URL%
echo Threads:      %LLAMA_THREADS%
echo Context:      %LLAMA_CTX_SIZE%
echo App URL:      http://127.0.0.1:8000
echo TTS:          %TTS_ENGINE% / %TTS_LANG% / %TTS_VOICE%
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
