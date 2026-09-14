@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
set PYTHONUTF8=1
title MuXVS
cd /d "%~dp0"

if not exist env\python.exe (
    echo Ошибка: виртуальное окружение не найдено.
    echo Сначала выполните 'run-MuXVS-installer.bat'.
    pause
    exit /b 1
)

set PYTHON=env\python.exe
set SCRIPT=app.py

call :check_internet_connection
call :running_interface
exit /b 0

:check_internet_connection
echo Проверяю подключение к интернету...
%PYTHON% -c "import socket; socket.create_connection(('huggingface.co', 443), timeout=5)" >nul 2>&1 && (
    echo Подключение к интернету есть.
    set "INTERNET_AVAILABLE=1"
    goto :check_end
)
echo Подключение к интернету не обнаружено.
set "INTERNET_AVAILABLE=0"
:check_end
echo.
exit /b 0

:running_interface
cls
echo ==== Запуск MuXVS ====

if not exist %SCRIPT% (
    echo Критическая ошибка: не найден основной скрипт %SCRIPT%!
    pause
    exit /b 1
)

if "%INTERNET_AVAILABLE%"=="0" (
    echo Запуск в ОФФЛАЙН-режиме...
    %PYTHON% %SCRIPT% --offline
) else (
    echo Запуск в ОНЛАЙН-режиме...
    %PYTHON% %SCRIPT%
)

if errorlevel 1 (
    echo Ошибка: не удалось запустить приложение (код ошибки: %errorlevel%)
    pause
    exit /b 1
)

exit /b 0
