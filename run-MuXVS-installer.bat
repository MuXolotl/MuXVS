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
echo MuXVS installed successfully!
echo To start MuXVS, run 'run-MuXVS.bat'.
echo.
pause
exit /b 0

:install_miniconda
if exist "%MINICONDA_DIR%\_conda.exe" (
    echo Miniconda is already installed. Skipping.
    exit /b 0
)

echo Miniconda was not found. Downloading and installing...
powershell -NoProfile -Command "& {Invoke-WebRequest -Uri '%MINICONDA_URL%' -OutFile 'miniconda.exe'}"
if not exist "miniconda.exe" goto :download_error

start /wait "" miniconda.exe /InstallationType=JustMe /RegisterPython=0 /S /D=%MINICONDA_DIR%
if errorlevel 1 goto :install_error

del miniconda.exe
echo Miniconda installed.
echo.
exit /b 0

:create_conda_env
cls
set "FORCE_CREATE="
if not exist "%ENV_DIR%\python.exe" goto :create_conda_env_new
for /f %%v in ('"%ENV_DIR%\python.exe" -c "import sys;print(f'{sys.version_info[0]}.{sys.version_info[1]}')"') do set "ENV_PY=%%v"
if "!ENV_PY!"=="%PYTHON_VERSION%" (
    echo A Python %PYTHON_VERSION% environment already exists. Skipping.
    echo.
    exit /b 0
)
echo The environment uses Python !ENV_PY!; recreating it for Python %PYTHON_VERSION%...
echo.
set "FORCE_CREATE=--force"

:create_conda_env_new
echo Creating Python environment (Python %PYTHON_VERSION%)...
call "%MINICONDA_DIR%\_conda.exe" create --no-shortcuts -y -k --prefix "%ENV_DIR%" python=%PYTHON_VERSION% !FORCE_CREATE!
if errorlevel 1 goto :error
echo Environment created.
echo.
exit /b 0

:install_dependencies
cls
echo Installing dependencies. This may take a while...

set "TORCH_BACKEND=cpu"
set "NVIDIA_GPUS=0"
for /f %%g in ('powershell -NoProfile -Command "(Get-CimInstance Win32_VideoController ^| Where-Object { $_.Name -match 'NVIDIA' } ^| Measure-Object).Count"') do set "NVIDIA_GPUS=%%g"
if not "!NVIDIA_GPUS!"=="0" set "TORCH_BACKEND=cu128"

echo  PyTorch: %TORCH_BACKEND%

"%ENV_DIR%\python.exe" -m pip install --upgrade pip setuptools
if errorlevel 1 goto :error

if /i "%TORCH_BACKEND%"=="cpu" (
    echo Installing PyTorch (CPU)...
    "%ENV_DIR%\python.exe" -m pip install torch==2.11 torchaudio==2.11
) else (
    echo Installing PyTorch (CUDA)...
    "%ENV_DIR%\python.exe" -m pip install torch==2.11 torchaudio==2.11 --index-url https://download.pytorch.org/whl/cu128
)
if errorlevel 1 goto :error

"%ENV_DIR%\python.exe" -m pip install torchcodec==0.11
if errorlevel 1 goto :error

"%ENV_DIR%\python.exe" -m pip install -r "%PRINCIPAL%\requirements.txt"
if errorlevel 1 goto :error

echo Dependencies installed.
echo.
exit /b 0

:download_ffmpeg
cls
echo Checking for ffmpeg and ffprobe...
if exist "%PRINCIPAL%\ffmpeg.exe" (
    if exist "%PRINCIPAL%\ffprobe.exe" (
        echo ffmpeg and ffprobe are already installed. Skipping.
        exit /b 0
    )
)

echo Downloading ffmpeg and ffprobe...
powershell -NoProfile -Command "& {Invoke-WebRequest -Uri 'https://huggingface.co/Politrees/RVC_resources/resolve/main/tools/ffmpeg/ffmpeg.exe?download=true' -OutFile 'ffmpeg.exe'}"
if not exist "ffmpeg.exe" goto :download_error

powershell -NoProfile -Command "& {Invoke-WebRequest -Uri 'https://huggingface.co/Politrees/RVC_resources/resolve/main/tools/ffmpeg/ffprobe.exe?download=true' -OutFile 'ffprobe.exe'}"
if not exist "ffprobe.exe" goto :download_error

echo ffmpeg and ffprobe downloaded.
echo.
exit /b 0

:verify_installation
cls
echo Checking the installation...
"%ENV_DIR%\python.exe" -c "import torch, numpy, scipy, sklearn, librosa, soundfile, gradio, faiss, omegaconf, edge_tts, gdown; from FlashSR.FlashSR import FlashSR; from TorchJaekwon.Util.UtilAudio import UtilAudio"
if errorlevel 1 goto :verify_error
echo All packages are available.
echo.

"%ENV_DIR%\python.exe" -c "import torch; print('CUDA_OK' if torch.cuda.is_available() else 'CUDA_NO')" | findstr /C:"CUDA_OK" >nul
if %errorlevel% equ 0 (
    echo CUDA is available; voice conversion will use the GPU.
) else (
    if /i not "%TORCH_BACKEND%"=="cpu" (
        echo WARNING: CUDA is unavailable, although an NVIDIA GPU was detected.
        echo   Update the driver from https://www.nvidia.com
    ) else (
        echo The application will run on the CPU (much slower).
    )
)
echo.
exit /b 0

:verify_error
echo.
echo Could not import some packages.
echo Run the installer again.
pause
exit /b 1

:download_error
echo.
echo Download failed. Check your internet connection and try again.
goto :error

:install_error
echo.
echo Could not install Miniconda.
goto :error

:error
echo.
echo An installation error occurred. See the output above for details.
pause
exit /b 1
