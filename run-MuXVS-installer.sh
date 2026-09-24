#!/bin/bash
# Установщик MuXVS (Linux / macOS).
set -euo pipefail

PYTHON_VERSION="3.11"

OS_NAME="$(uname)"
OS_ARCH="$(uname -m)"

# ── Устройство для PyTorch ─────────────────────────────────────
TORCH_DEVICE="cpu"
if [ "$OS_NAME" = "Linux" ] && command -v nvidia-smi > /dev/null 2>&1; then
    TORCH_DEVICE="cuda"
fi

echo "============================================="
echo "  MuXVS Installer"
echo "  Python: $PYTHON_VERSION | Устройство: $TORCH_DEVICE"
echo "============================================="
echo

PRINCIPAL="$(pwd)"
MINICONDA_DIR="$HOME/miniconda3"
ENV_DIR="$PRINCIPAL/env"

# ── URL установщика Miniconda ───────────────────────────────────
MINICONDA_BASE="https://repo.anaconda.com/miniconda"
MINICONDA_URL=""

case "$OS_NAME" in
    Darwin)
        if [ "$OS_ARCH" = "arm64" ]; then
            MINICONDA_URL="$MINICONDA_BASE/Miniconda3-py311_26.7.1-1-MacOSX-arm64.sh"
        else
            MINICONDA_URL="$MINICONDA_BASE/Miniconda3-py311_26.7.1-1-MacOSX-x86_64.sh"
        fi
        ;;
    Linux)
        if [ "$OS_ARCH" = "x86_64" ]; then
            MINICONDA_URL="$MINICONDA_BASE/Miniconda3-py311_26.7.1-1-Linux-x86_64.sh"
        else
            echo "Ошибка: автоматическая установка Miniconda поддерживается только на x86_64 Linux."
            echo "Установите Miniconda/conda вручную и запустите скрипт повторно."
            exit 1
        fi
        ;;
    *)
        echo "Ошибка: неподдерживаемая ОС: $OS_NAME"
        exit 1
        ;;
esac

install_miniconda() {
    if [ -x "$MINICONDA_DIR/bin/conda" ] || [ -x "$HOME/.conda/bin/conda" ]; then
        if [ -x "$MINICONDA_DIR/bin/conda" ]; then
            CONDA_EXE="$MINICONDA_DIR/bin/conda"
        else
            CONDA_EXE="$HOME/.conda/bin/conda"
        fi
        echo "Conda уже установлена ($CONDA_EXE) — пропускаю."
        return
    fi

    echo "Miniconda не найдена. Скачиваю и устанавливаю..."
    curl -fSL -o miniconda.sh "$MINICONDA_URL"
    bash miniconda.sh -b -p "$MINICONDA_DIR"
    rm -f miniconda.sh
    CONDA_EXE="$MINICONDA_DIR/bin/conda"
    echo "Miniconda установлена."
    echo
}

create_conda_env() {
    echo "Создаю Python-окружение (python=$PYTHON_VERSION) в $ENV_DIR ..."
    "$CONDA_EXE" create --yes --prefix "$ENV_DIR" "python=$PYTHON_VERSION"
}

install_dependencies() {
    echo "Устанавливаю зависимости (это может занять заметное время)..."
    local env_python="$ENV_DIR/bin/python"

    "$env_python" -m pip install --upgrade pip setuptools

    if [ "$TORCH_DEVICE" = "cuda" ]; then
        echo "Устанавливаю PyTorch (CUDA, cu128)..."
        "$env_python" -m pip install torch==2.11.0 torchaudio==2.11.0 \
            --index-url https://download.pytorch.org/whl/cu128
    else
        echo "Устанавливаю PyTorch (CPU/MPS)..."
        "$env_python" -m pip install torch==2.11.0 torchaudio==2.11.0
    fi

    "$env_python" -m pip install -r "$PRINCIPAL/requirements.txt"
    echo "Зависимости установлены."
    echo
}

install_ffmpeg() {
    if command -v ffmpeg > /dev/null 2>&1 && command -v ffprobe > /dev/null 2>&1; then
        echo "FFmpeg уже есть в системе — пропускаю."
        return
    fi

    if command -v brew > /dev/null 2>&1; then
        echo "Устанавливаю FFmpeg через Homebrew..."
        brew install ffmpeg
    elif command -v apt-get > /dev/null 2>&1; then
        echo "Устанавливаю FFmpeg через apt..."
        sudo apt-get update && sudo apt-get install -y ffmpeg
    elif command -v pacman > /dev/null 2>&1; then
        echo "Устанавливаю FFmpeg через pacman..."
        sudo pacman -Syu --noconfirm ffmpeg
    elif command -v dnf > /dev/null 2>&1; then
        echo "Устанавливаю FFmpeg через dnf..."
        sudo dnf install -y ffmpeg --allowerasing || install_ffmpeg_flatpak
    else
        echo "Пакетный менеджер не найден. Пробую Flatpak..."
        install_ffmpeg_flatpak
    fi
}

install_ffmpeg_flatpak() {
    if ! command -v flatpak > /dev/null 2>&1; then
        echo "Ошибка: Flatpak не установлен. Установите FFmpeg вручную и повторите."
        exit 1
    fi
    flatpak install --user -y flathub org.freedesktop.Platform.ffmpeg
}

install_miniconda

if [ -x "$ENV_DIR/bin/python" ] && "$ENV_DIR/bin/python" --version 2>&1 | grep -q "Python $PYTHON_VERSION"; then
    echo "Окружение $ENV_DIR с Python $PYTHON_VERSION уже существует — пропускаю создание."
    echo
else
    create_conda_env
fi

install_dependencies
install_ffmpeg

echo
echo "MuXVS установлен успешно!"
echo "Для запуска выполните: ./run-MuXVS.sh"
echo
