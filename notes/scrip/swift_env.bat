@echo off
setlocal
title swift training - swift_env

rem ---------------------------------------------------------------
rem  Activates the dedicated ms-swift training env on D: and opens a shell.
rem ---------------------------------------------------------------

cd /d "%~dp0"

rem Clear the broken socks5 proxy on 127.0.0.1:1080.
set "HTTP_PROXY="
set "HTTPS_PROXY="
set "ALL_PROXY="
set "http_proxy="
set "https_proxy="
set "all_proxy="

rem Keep pip caches off the C: drive.
set "PIP_CACHE_DIR=D:\pip-cache"

rem ---------------------------------------------------------------
set "CONDA_ROOT=D:\ProgramData\anaconda3"
set "ENV_DIR=D:\conda_envs\swift_env"

if not exist "%ENV_DIR%\python.exe" (
    echo [ERROR] conda env not found: %ENV_DIR%
    echo         Run: conda create -p D:\conda_envs\swift_env python=3.10
    pause
    exit /b 1
)

call "%CONDA_ROOT%\Scripts\activate.bat" "%ENV_DIR%"
if errorlevel 1 (
    echo [ERROR] Failed to activate %ENV_DIR%
    pause
    exit /b 1
)

echo ============================================================
echo  Python : %CONDA_PREFIX%\python.exe
echo  Env    : %CONDA_PREFIX%
echo  Workdir: %CD%
echo ============================================================
echo.
echo Example (LoRA SFT on minimind-3):
echo   swift sft --model base_models\minimind-3 --model_type qwen3 --template qwen3
echo             --tuner_type lora --dataset dataset\train.jsonl --output_dir out
echo.
echo Env activated. Type "exit" to close this window.
echo.

cmd /k
endlocal
