@echo off
setlocal
title model_train - MiniMind

rem ---------------------------------------------------------------
rem  One-click launcher: activate the conda env on D: and open Jupyter
rem ---------------------------------------------------------------

rem Work from the folder this script lives in.
cd /d "%~dp0"

rem Clear the broken socks5 proxy on 127.0.0.1:1080.
rem Delete these lines once a working proxy is available.
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
set "ENV_DIR=D:\conda_envs\model_train_env"

if not exist "%ENV_DIR%\python.exe" (
    echo [ERROR] conda env not found: %ENV_DIR%
    echo         Run: conda create -p D:\conda_envs\model_train_env python=3.10
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

where jupyter >nul 2>nul
if errorlevel 1 (
    echo [ERROR] jupyter is not installed in this environment
    echo         Run: pip install notebook -i https://pypi.tuna.tsinghua.edu.cn/simple
    pause
    exit /b 1
)

echo Starting Jupyter Notebook...
jupyter notebook
if errorlevel 1 (
    echo.
    echo [ERROR] jupyter notebook exited abnormally
    pause
)

endlocal
