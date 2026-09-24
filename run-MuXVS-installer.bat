@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
title MuXVS Installer
cd /d "%~dp0"

echo =============================================
echo  MuXVS Installer
echo =============================================
echo.

set "PYTHON_VERSION=3.11"
set "PRINCIPAL=%cd%"
set "MINICONDA_DIR=%UserProfile%\Miniconda3"
set "ENV_DIR=%PRINCIPAL%\env"
set "MINICONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-py311_26.7.1-1-Windows-x86_64.exe"

call :install_miniconda
if errorlevel 1 goto :error
call :create_conda_env
if errorlevel 1 goto :error
call :install_dependencies
if errorlevel 1 goto :error
call :download_ffmpeg
if errorlevel 1 goto :error
call :verify_installation
if errorlevel 1 goto :error

cls
echo.
echo MuXVS установлен успешно!
echo Для запуска MuXVS выполните 'run-MuXVS.bat'.
echo.
pause
exit /b 0

:install_miniconda
if exist "%MINICONDA_DIR%\_conda.exe" (
    echo Miniconda уже установлен - пропускаю.
    exit /b 0
)

echo Miniconda не найден. Скачиваю и устанавливаю...
powershell -NoProfile -Command "& {Invoke-WebRequest -Uri '%MINICONDA_URL%' -OutFile 'miniconda.exe'}"
if not exist "miniconda.exe" goto :download_error

start /wait "" miniconda.exe /InstallationType=JustMe /RegisterPython=0 /S /D=%MINICONDA_DIR%
if errorlevel 1 goto :install_error

del miniconda.exe
echo Miniconda установлен.
echo.
exit /b 0

:create_conda_env
cls
set "FORCE_CREATE="
if exist "%ENV_DIR%\python.exe" (
    for /f %%v in ('"%ENV_DIR%\python.exe" -c "import sys;print(f'{sys.version_info[0]}.{sys.version_info[1]}')"') do set "ENV_PY=%%v"
    if "!ENV_PY!"=="%PYTHON_VERSION%" (
        echo Окружение с Python %PYTHON_VERSION% уже существует - пропускаю.
        echo.
        exit /b 0
    )
    echo Окружение создано для Python !ENV_PY! - пересоздаю под Python %PYTHON_VERSION%...
    echo.
    set "FORCE_CREATE=--force"
)

echo Создаю Python-окружение (python=%PYTHON_VERSION%)...
call "%MINICONDA_DIR%\_conda.exe" create --no-shortcuts -y -k --prefix "%ENV_DIR%" python=%PYTHON_VERSION% !FORCE_CREATE!
if errorlevel 1 goto :error
echo Окружение создано.
echo.
exit /b 0

:install_dependencies
cls
echo Устанавливаю зависимости (это может занять какое-то время)...

set "TORCH_BACKEND=cpu"
set "NVIDIA_GPUS=0"
for /f %%g in ('powershell -NoProfile -Command "(Get-CimInstance Win32_VideoController ^| Where-Object { $_.Name -match 'NVIDIA' } ^| Measure-Object).Count"') do set "NVIDIA_GPUS=%%g"
if not "!NVIDIA_GPUS!"=="0" set "TORCH_BACKEND=cu128"

echo  PyTorch: %TORCH_BACKEND%

"%ENV_DIR%\python.exe" -m pip install --upgrade pip setuptools
if errorlevel 1 goto :error

if /i "%TORCH_BACKEND%"=="cpu" (
    echo Устанавливаю PyTorch (CPU)...
    "%ENV_DIR%\python.exe" -m pip install torch==2.11 torchaudio==2.11
) else (
    echo Устанавливаю PyTorch (CUDA)...
    "%ENV_DIR%\python.exe" -m pip install torch==2.11 torchaudio==2.11 --index-url https://download.pytorch.org/whl/cu128
)
if errorlevel 1 goto :error

"%ENV_DIR%\python.exe" -m pip install torchcodec==0.11
if errorlevel 1 goto :error

"%ENV_DIR%\python.exe" -m pip install -r "%PRINCIPAL%\requirements.txt"
if errorlevel 1 goto :error

echo Зависимости установлены.
echo.
exit /b 0

:download_ffmpeg
cls
echo Проверяю ffmpeg и ffprobe...
if exist "%PRINCIPAL%\ffmpeg.exe" (
    if exist "%PRINCIPAL%\ffprobe.exe" (
        echo ffmpeg и ffprobe уже есть - пропускаю.
        exit /b 0
    )
)

echo Скачиваю ffmpeg и ffprobe...
powershell -NoProfile -Command "& {Invoke-WebRequest -Uri 'https://huggingface.co/Politrees/RVC_resources/resolve/main/tools/ffmpeg/ffmpeg.exe?download=true' -OutFile 'ffmpeg.exe'}"
if not exist "ffmpeg.exe" goto :download_error

powershell -NoProfile -Command "& {Invoke-WebRequest -Uri 'https://huggingface.co/Politrees/RVC_resources/resolve/main/tools/ffmpeg/ffprobe.exe?download=true' -OutFile 'ffprobe.exe'}"
if not exist "ffprobe.exe" goto :download_error

echo ffmpeg и ffprobe скачаны.
echo.
exit /b 0

:verify_installation
cls
echo Проверяю установку...
"%ENV_DIR%\python.exe" -c "import torch, numpy, scipy, sklearn, librosa, soundfile, gradio, faiss, omegaconf, edge_tts, gdown; from FlashSR.FlashSR import FlashSR; from TorchJaekwon.Util.UtilAudio import UtilAudio"
if errorlevel 1 goto :verify_error
echo Все пакеты на месте.
echo.

"%ENV_DIR%\python.exe" -c "import torch; print('CUDA_OK' if torch.cuda.is_available() else 'CUDA_NO')" | findstr /C:"CUDA_OK" >nul
if %errorlevel% equ 0 (
    echo CUDA доступна - преобразование голоса будет работать на GPU.
) else (
    if /i not "%TORCH_BACKEND%"=="cpu" (
        echo ВНИМАНИЕ: CUDA недоступна, хотя видеокарта NVIDIA обнаружена.
        echo   Обновите драйвер с https://www.nvidia.com
    ) else (
        echo Программа будет работать на CPU (значительно медленнее).
    )
)
echo.
exit /b 0

:verify_error
echo.
echo Не удалось импортировать некоторые пакеты.
echo Повторите запуск инсталлятора.
pause
exit /b 1

:download_error
echo.
echo Ошибка скачивания. Проверьте подключение к интернету и повторите.
goto :error

:install_error
echo.
echo Не удалось установить Miniconda.
goto :error

:error
echo.
echo Во время установки произошла ошибка. Подробности - в выводе выше.
pause
exit /b 1
