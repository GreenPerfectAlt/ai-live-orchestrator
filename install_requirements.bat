@echo off
setlocal EnableExtensions
set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%" || exit /b 1

if not exist "models" mkdir "models"
if not exist "models\silero" mkdir "models\silero"
if not exist "models\supertonic3" mkdir "models\supertonic3"

if not exist ".venv\Scripts\python.exe" (
  echo [SETUP] Creating .venv...
  where py >nul 2>nul
  if not errorlevel 1 (
    py -3.12 -m venv ".venv" >nul 2>nul
    if not exist ".venv\Scripts\python.exe" py -3.11 -m venv ".venv" >nul 2>nul
  )
  if not exist ".venv\Scripts\python.exe" (
    python -m venv ".venv"
  )
)

call ".venv\Scripts\activate.bat" || (
  echo [ERROR] Cannot activate .venv
  pause
  exit /b 1
)

python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
if errorlevel 1 goto fail

python -c "import torch" >nul 2>nul
if errorlevel 1 (
  echo [SETUP] Installing PyTorch CPU for Silero...
  python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
  if errorlevel 1 (
    echo [SETUP] CPU wheel failed. Trying normal PyPI torch...
    python -m pip install torch
    if errorlevel 1 goto fail
  )
)

echo.
echo [OK] Dependencies installed.
echo.
echo Next:
echo 1. Put llama-server.exe into this folder OR edit run_llama*.bat path.
echo 2. Put GGUF + mmproj into models\
echo 3. Run needed run_llama*.bat
pause
exit /b 0

:fail
echo.
echo [ERROR] Install failed.
echo Use Python 3.11 or 3.12. Check internet.
pause
exit /b 1
