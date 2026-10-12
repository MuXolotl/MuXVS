@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
set "PYTHONUTF8=1"
title MuXVS
cd /d "%~dp0"

if not exist "env\python.exe" (
    echo Error: the Python environment was not found.
    echo Run 'run-MuXVS-installer.bat' first.
    pause
    exit /b 1
)

set "PYTHON=env\python.exe"
set "SCRIPT=app.py"

call :check_internet_connection
call :running_interface %*
exit /b %errorlevel%

:check_internet_connection
echo Checking the internet connection...
"%PYTHON%" -c "import socket; socket.create_connection(('huggingface.co', 443), timeout=5)" >nul 2>&1
if errorlevel 1 goto :connection_unavailable
echo Internet connection is available.
set "INTERNET_AVAILABLE=1"
goto :check_end

:connection_unavailable
echo No internet connection detected.
set "INTERNET_AVAILABLE=0"

:check_end
echo.
exit /b 0

:running_interface
cls
echo ==== Starting MuXVS ====

if not exist "%SCRIPT%" (
    echo Critical error: main script %SCRIPT% was not found.
    pause
    exit /b 1
)

if "%INTERNET_AVAILABLE%"=="0" (
    echo Starting in OFFLINE mode...
    "%PYTHON%" "%SCRIPT%" --offline %*
) else (
    echo Starting in ONLINE mode...
    "%PYTHON%" "%SCRIPT%" %*
)

set "APP_EXIT_CODE=!errorlevel!"
if errorlevel 1 goto :application_error
exit /b 0

:application_error
echo Error: the application failed to start (exit code !APP_EXIT_CODE!).
pause
exit /b 1
