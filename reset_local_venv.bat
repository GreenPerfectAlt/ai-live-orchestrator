@echo off
setlocal EnableExtensions
set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%" || exit /b 1
@echo off
setlocal
set "PARLOR_VENV_DIR=%LOCALAPPDATA%\ParlorJarvis\venv-llamacpp"
echo This will delete the local Parlor Python venv on this PC:
echo %PARLOR_VENV_DIR%
echo.
choice /C YN /M "Delete it"
if errorlevel 2 exit /b 0
rmdir /s /q "%PARLOR_VENV_DIR%"
echo Deleted. Run one of run_llama_*_flash.bat again.
pause

set "SILERO_MODEL=v5_5_ru"
set "SILERO_MODEL_URL=https://models.silero.ai/models/tts/ru/v5_5_ru.pt"
