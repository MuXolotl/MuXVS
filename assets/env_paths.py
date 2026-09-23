"""Пути хранения файлов обучения в зависимости от среды запуска.

Локально всё лежит в папке репозитория; в облачных блокнотах код и обучаемые
модели хранятся раздельно (диск блокнота эфемерен, модели — в постоянном
хранилище). Касается только обучения: остальное всегда пишется в репозиторий.

Colab повторяет схему PolTrain_Colab.ipynb: код в /content/, модели в
SAVE_DIR на Google Drive. Kaggle — схему PolTrain_kaggle.ipynb: код
в /kaggle/working/, модели в SAVE_DIR там же.
"""

import os

from assets.notebook_check import colab_check, kaggle_check

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COLAB_DRIVE_MOUNT = "/content/drive"
COLAB_TRAINING_DIR = "/content/drive/MyDrive/MuXVS"
KAGGLE_TRAINING_DIR = "/kaggle/working/Models"


def training_logs_dir() -> str:
    """Корень экспериментов обучения для текущей среды. Только путь, без проверок."""
    if colab_check():
        return COLAB_TRAINING_DIR
    if kaggle_check():
        return KAGGLE_TRAINING_DIR
    return os.path.join(PROJECT_ROOT, "logs")


def is_colab_drive_mounted() -> bool:
    """Подключён ли Google Drive (во избежание записи в локальную папку-тень)."""
    return os.path.ismount(COLAB_DRIVE_MOUNT)


def require_training_logs_dir() -> str:
    """Корень экспериментов для действий: проверяет Drive и создаёт папку."""
    path = training_logs_dir()
    if colab_check() and not is_colab_drive_mounted():
        raise RuntimeError("Google Drive не подключён: выполните drive.mount('/content/drive') в блокноте и повторите.")
    os.makedirs(path, exist_ok=True)
    return path
