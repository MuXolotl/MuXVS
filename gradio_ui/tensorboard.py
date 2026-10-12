"""Встроенный TensorBoard: запуск в фоне и ссылка с закреплёнными графиками.

Как в Applio: сервер поднимается внутри процесса интерфейса (`tensorboard.program`)
и открывается через iframe прямо во вкладке обучения. Повторный запуск возвращает
уже поднятый экземпляр. Импорт tensorboard ленивый — без него интерфейс работает,
а кнопка показывает ошибку.
"""

import importlib.machinery
import importlib.util
import json
import os
import sys
import threading
import types
import urllib.parse
from pathlib import Path

DEFAULT_TENSORBOARD_PORT = 6006

# Теги скаляров, закрепляемые в открытом TensorBoard (пишет rvc/training/train.py).
PINNED_TAGS = ["loss/g/mel", "loss/g/total"]

# Держим запущенные серверы, чтобы их не собрал GC раньше процесса интерфейса.
_TENSORBOARD_SERVERS = []
_TENSORBOARD_URL = None
_TENSORBOARD_LOCK = threading.Lock()


def _pinned_url(url: str, tags) -> str:
    """Ссылка на TensorBoard с закреплёнными карточками нужных скаляров."""
    cards = [{"plugin": "scalars", "tag": tag} for tag in tags]
    query = urllib.parse.quote(json.dumps(cards, separators=(",", ":")))
    return f"{url.rstrip('/')}/?pinnedCards={query}"


def _force_tensorboard_stub() -> None:
    """TensorBoard работает на своём заглушечном TF, не импортируя настоящий.

    При старте TensorBoard делает `import tensorflow` лишь ради проверки его
    наличия (`tensorboard.compat.tf`). MuXVS TensorFlow не использует, а сам
    импорт в свежих окружениях сыплет мусором вида «Could not find cuda
    drivers» (образы Colab переехали на CUDA 13, а TF собран под CUDA 12;
    фильтры логов вроде TF_CPP_MIN_LOG_LEVEL этот спам в новых TF не глушат).

    Штатный выключатель — модуль `tensorboard.compat.notf`: при его наличии
    сервер работает на `tensorflow_stub`. Создаём файл в установке пакета —
    это покрывает и дочерние процессы тензорборда; если установка только для
    чтения, подменяем модуль хотя бы для здешнего запуска в этом процессе.
    """
    try:
        if importlib.util.find_spec("tensorboard.compat.notf"):
            return
    except ImportError:
        return  # tensorboard не установлен — глушить нечего
    except ValueError:
        return  # уже подменён в sys.modules

    import tensorboard

    notf_path = Path(tensorboard.__file__).parent / "compat" / "notf.py"
    try:
        if not notf_path.exists():
            notf_path.write_text("# Force tensorflow_stub: MuXVS does not use TensorFlow.\n", encoding="utf-8")
    except OSError:
        module = types.ModuleType("tensorboard.compat.notf")
        module.__spec__ = importlib.machinery.ModuleSpec("tensorboard.compat.notf", loader=None)
        sys.modules.setdefault("tensorboard.compat.notf", module)


def launch_tensorboard(logs_dir: str, port: int = DEFAULT_TENSORBOARD_PORT) -> str:
    """Запускает TensorBoard в фоновом потоке.

    Возвращает ссылку с закреплёнными карточками или строку 'Ошибка...'.
    Сервер живёт столько же, сколько процесс интерфейса.
    """
    global _TENSORBOARD_URL

    os.makedirs(logs_dir, exist_ok=True)

    with _TENSORBOARD_LOCK:
        if _TENSORBOARD_URL:
            return _TENSORBOARD_URL
        try:
            _force_tensorboard_stub()
            from tensorboard import program

            board = program.TensorBoard()
            board.configure(argv=[None, "--logdir", logs_dir, "--port", str(port), "--path_prefix", "/tensorboard"])
            url = board.launch()
        except Exception as error:
            return f"Ошибка запуска TensorBoard: {error}"
        _TENSORBOARD_SERVERS.append(board)
        _TENSORBOARD_URL = _pinned_url(url, PINNED_TAGS)
        return _TENSORBOARD_URL
