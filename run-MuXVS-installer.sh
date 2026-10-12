#!/bin/bash
# MuXVS installer for Linux and macOS.
set -euo pipefail

PYTHON_VERSION="3.11"

OS_NAME="$(uname)"
OS_ARCH="$(uname -m)"

# -- PyTorch device ---------------------------------------------------
TORCH_DEVICE="cpu"
if [ "$OS_NAME" = "Linux" ] && command -v nvidia-smi > /dev/null 2>&1; then
    TORCH_DEVICE="cuda"
fi

echo "============================================="
echo "  MuXVS Installer"
echo "  Python: $PYTHON_VERSION | Device: $TORCH_DEVICE"
echo "============================================="
echo

PRINCIPAL="$(pwd)"
MINICONDA_DIR="$HOME/miniconda3"
ENV_DIR="$PRINCIPAL/env"

# -- Miniconda installer URL -----------------------------------------
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
            echo "Error: automatic Miniconda installation is supported only on x86_64 Linux."
            echo "Install Miniconda/conda manually, then run this script again."
            exit 1
        fi
        ;;
    *)
        echo "Error: unsupported operating system: $OS_NAME"
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
        echo "Conda is already installed ($CONDA_EXE). Skipping."
        return
    fi

    echo "Miniconda was not found. Downloading and installing..."
    curl -fSL -o miniconda.sh "$MINICONDA_URL"
    bash miniconda.sh -b -p "$MINICONDA_DIR"
    rm -f miniconda.sh
    CONDA_EXE="$MINICONDA_DIR/bin/conda"
    echo "Miniconda installed."
    echo
}

create_conda_env() {
    echo "Creating Python environment (Python=$PYTHON_VERSION) at $ENV_DIR..."
    "$CONDA_EXE" create --yes --prefix "$ENV_DIR" "python=$PYTHON_VERSION"
}

install_dependencies() {
    echo "Installing dependencies. This may take a while..."
    local env_python="$ENV_DIR/bin/python"

    "$env_python" -m pip install --upgrade pip setuptools

    if [ "$TORCH_DEVICE" = "cuda" ]; then
        echo "Installing PyTorch (CUDA, cu128)..."
        "$env_python" -m pip install torch==2.11 torchaudio==2.11 torchcodec==0.11 \
            --index-url https://download.pytorch.org/whl/cu128
    elif [ "$OS_NAME" = "Linux" ]; then
        echo "Installing PyTorch (CPU)..."
        "$env_python" -m pip install torch==2.11 torchaudio==2.11 torchcodec==0.11 \
            --index-url https://download.pytorch.org/whl/cpu
    else
        echo "Installing PyTorch (CPU/MPS)..."
        "$env_python" -m pip install torch==2.11 torchaudio==2.11 torchcodec==0.11
    fi

    "$env_python" -m pip install -r "$PRINCIPAL/requirements.txt"
    echo "Dependencies installed."
    echo
}

install_ffmpeg() {
    if command -v ffmpeg > /dev/null 2>&1 && command -v ffprobe > /dev/null 2>&1; then
        echo "FFmpeg is already installed on the system. Skipping."
        return
    fi

    if command -v brew > /dev/null 2>&1; then
        echo "Installing FFmpeg with Homebrew..."
        brew install ffmpeg
    elif command -v apt-get > /dev/null 2>&1; then
        echo "Installing FFmpeg with apt..."
        sudo apt-get update && sudo apt-get install -y ffmpeg
    elif command -v pacman > /dev/null 2>&1; then
        echo "Installing FFmpeg with pacman..."
        sudo pacman -Syu --noconfirm ffmpeg
    elif command -v dnf > /dev/null 2>&1; then
        echo "Installing FFmpeg with dnf..."
        sudo dnf install -y ffmpeg --allowerasing || install_ffmpeg_flatpak
    else
        echo "No supported package manager was found. Trying Flatpak..."
        install_ffmpeg_flatpak
    fi
}

install_ffmpeg_flatpak() {
    if ! command -v flatpak > /dev/null 2>&1; then
        echo "Error: Flatpak is not installed. Install FFmpeg manually and run this script again."
        exit 1
    fi
    flatpak install --user -y flathub org.freedesktop.Platform.ffmpeg
}

install_miniconda

if [ -x "$ENV_DIR/bin/python" ] && "$ENV_DIR/bin/python" --version 2>&1 | grep -q "Python $PYTHON_VERSION"; then
    echo "The Python $PYTHON_VERSION environment at $ENV_DIR already exists. Skipping creation."
    echo
else
    create_conda_env
fi

install_dependencies
install_ffmpeg

echo
echo "MuXVS installed successfully!"
echo "Run the application with: ./run-MuXVS.sh"
echo
