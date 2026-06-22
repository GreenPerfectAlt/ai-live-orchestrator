@echo off
setlocal EnableExtensions

set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%" || (
  echo [ERROR] Cannot enter project folder: %PROJECT_ROOT%
  pause
  exit /b 1
)

set "VENV_PY=.venv\Scripts\python.exe"
set "VENV_ACT=.venv\Scripts\activate.bat"

call :create_folders

:menu
cls
echo.
echo AI Live Orchestrator - Setup Menu
echo =================================
echo.
call :status_short
echo.
echo [1] Install base app dependencies
echo [2] Install PyTorch CPU for Silero RU
echo [3] Full install: base deps + PyTorch CPU
echo [4] Check environment
echo [5] Reset .venv
echo [6] Create folders
echo [0] Exit
echo.
choice /C 1234560 /N /M "Select option: "

if errorlevel 7 goto end
if errorlevel 6 call :create_folders & pause & goto menu
if errorlevel 5 call :reset_venv & pause & goto menu
if errorlevel 4 call :check_environment & pause & goto menu
if errorlevel 3 call :full_install & pause & goto menu
if errorlevel 2 call :install_torch_cpu & pause & goto menu
if errorlevel 1 call :install_base & pause & goto menu

goto menu

:find_python
set "PY_CMD="

where py >nul 2>nul
if not errorlevel 1 (
  py -3.12 -c "import sys; sys.exit(0 if sys.version_info[:2] == (3,12) else 1)" >nul 2>nul
  if not errorlevel 1 set "PY_CMD=py -3.12"

  if not defined PY_CMD (
    py -3.11 -c "import sys; sys.exit(0 if sys.version_info[:2] == (3,11) else 1)" >nul 2>nul
    if not errorlevel 1 set "PY_CMD=py -3.11"
  )
)

if not defined PY_CMD (
  python -c "import sys; sys.exit(0 if sys.version_info[:2] in [(3,11),(3,12)] else 1)" >nul 2>nul
  if not errorlevel 1 set "PY_CMD=python"
)

if not defined PY_CMD exit /b 1
exit /b 0

:create_folders
if not exist "models" mkdir "models"
if not exist "models\silero" mkdir "models\silero"
if not exist "models\supertonic3" mkdir "models\supertonic3"
if not exist "models\.hf_cache" mkdir "models\.hf_cache"
if not exist "models\.torch" mkdir "models\.torch"
echo [OK] Folders ready.
exit /b 0

:ensure_venv
if exist "%VENV_PY%" exit /b 0

call :find_python
if errorlevel 1 (
  echo [ERROR] Python 3.11 or 3.12 not found.
  echo Install Python 3.11 or 3.12 and try again.
  exit /b 1
)

echo [SETUP] Creating .venv using %PY_CMD%...
%PY_CMD% -m venv ".venv"
if errorlevel 1 (
  echo [ERROR] Failed to create .venv.
  exit /b 1
)

if not exist "%VENV_PY%" (
  echo [ERROR] .venv was not created correctly.
  exit /b 1
)

exit /b 0

:activate_venv
call :ensure_venv
if errorlevel 1 exit /b 1

call "%VENV_ACT%" || (
  echo [ERROR] Cannot activate .venv.
  exit /b 1
)

exit /b 0

:install_base
echo.
echo [INSTALL] Base app dependencies
echo -------------------------------
if not exist "requirements.txt" (
  echo [ERROR] requirements.txt not found.
  exit /b 1
)

call :activate_venv
if errorlevel 1 exit /b 1

python -m pip install -U pip setuptools wheel
if errorlevel 1 goto install_fail

python -m pip install -r requirements.txt
if errorlevel 1 goto install_fail

echo.
echo [OK] Base dependencies installed.
echo.
echo Next:
echo 1. Put llama-server.exe into this folder OR edit run_llama_CUSTOM_TEMPLATE.bat.
echo 2. Put GGUF model into models\
echo 3. Optional: put mmproj*.gguf into models\ for vision/multimodal mode.
echo 4. Edit USER CONFIG - REQUIRED in run_llama_CUSTOM_TEMPLATE.bat.
echo 5. Run run_llama_CUSTOM_TEMPLATE.bat.
exit /b 0

:install_torch_cpu
echo.
echo [INSTALL] PyTorch CPU for Silero RU
echo -----------------------------------
call :activate_venv
if errorlevel 1 exit /b 1

python -c "import torch" >nul 2>nul
if not errorlevel 1 (
  echo [OK] torch already installed.
  exit /b 0
)

python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
if errorlevel 1 (
  echo [WARN] CPU wheel failed. Trying normal PyPI torch...
  python -m pip install torch
  if errorlevel 1 goto install_fail
)

echo [OK] PyTorch installed.
exit /b 0

:full_install
echo.
echo [INSTALL] Full setup
echo --------------------
call :install_base
if errorlevel 1 exit /b 1

call :install_torch_cpu
if errorlevel 1 exit /b 1

echo.
echo [OK] Full install complete.
exit /b 0

:reset_venv
echo.
echo [RESET] Delete .venv
echo --------------------
if not exist ".venv" (
  echo [OK] .venv does not exist.
  exit /b 0
)

choice /C YN /M "Delete .venv"
if errorlevel 2 (
  echo [OK] Canceled.
  exit /b 0
)

rmdir /s /q ".venv"
if exist ".venv" (
  echo [ERROR] Failed to delete .venv.
  exit /b 1
)

echo [OK] .venv deleted.
exit /b 0

:status_short
call :find_python >nul 2>nul
if defined PY_CMD (
  echo Python: %PY_CMD%
) else (
  echo Python: missing 3.11/3.12
)

if exist "%VENV_PY%" (
  echo .venv: found
) else (
  echo .venv: missing
)

if exist "llama-server.exe" (
  echo llama-server.exe: found
) else (
  echo llama-server.exe: missing
)

dir /b "models\*.gguf" >nul 2>nul
if not errorlevel 1 (
  echo GGUF models: found
) else (
  echo GGUF models: missing
)
exit /b 0

:check_environment
echo.
echo [CHECK] Environment
echo -------------------

call :find_python >nul 2>nul
if defined PY_CMD (
  echo [OK] Python command: %PY_CMD%
  %PY_CMD% --version
) else (
  echo [MISSING] Python 3.11 or 3.12
)

if exist "%VENV_PY%" (
  echo [OK] .venv
  "%VENV_PY%" --version
) else (
  echo [MISSING] .venv
)

if exist "requirements.txt" (
  echo [OK] requirements.txt
) else (
  echo [MISSING] requirements.txt
)

if exist "run_llama_CUSTOM_TEMPLATE.bat" (
  echo [OK] run_llama_CUSTOM_TEMPLATE.bat
) else (
  echo [MISSING] run_llama_CUSTOM_TEMPLATE.bat
)

if exist "llama-server.exe" (
  echo [OK] llama-server.exe
) else (
  echo [MISSING] llama-server.exe
)

if exist "models" (
  echo [OK] models\
) else (
  echo [MISSING] models\
)

dir /b "models\*.gguf" >nul 2>nul
if not errorlevel 1 (
  echo [OK] GGUF model files found:
  dir /b "models\*.gguf"
) else (
  echo [MISSING] models\*.gguf
)

dir /b "models\mmproj*.gguf" >nul 2>nul
if not errorlevel 1 (
  echo [OK] mmproj files found:
  dir /b "models\mmproj*.gguf"
) else (
  echo [OPTIONAL] models\mmproj*.gguf missing
)

if exist "%VENV_PY%" (
  echo.
  echo [CHECK] Python packages
  "%VENV_PY%" -c "import fastapi, uvicorn, numpy, requests; print('[OK] base packages')" 2>nul
  if errorlevel 1 echo [MISSING] base packages

  "%VENV_PY%" -c "import supertonic; print('[OK] supertonic')" 2>nul
  if errorlevel 1 echo [MISSING] supertonic

  "%VENV_PY%" -c "import torch; print('[OK] torch')" 2>nul
  if errorlevel 1 echo [OPTIONAL] torch missing / needed for Silero RU
)

echo.
echo [DONE] Check complete.
exit /b 0

:install_fail
echo.
echo [ERROR] Install failed.
echo Check internet connection and Python version.
exit /b 1

:end
exit /b 0
