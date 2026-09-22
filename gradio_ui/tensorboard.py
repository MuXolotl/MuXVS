"""Встроенный TensorBoard: запуск в фоне и ссылка с закреплёнными графиками.

Как в Applio: сервер поднимается внутри процесса интерфейса (`tensorboard.program`)
и открывается через iframe прямо во вкладке обучения. Повторный запуск возвращает
уже поднятый экземпляр. Импорт tensorboard ленивый — без него интерфейс работает,
а кнопка показывает ошибку.
"""

import json
import os
import threading
import urllib.parse

DEFAULT_TENSORBOARD_PORT = 6006

# Теги скаляров, закрепляемые в открытом TensorBoard (пишет rvc/training/train.py).
PINNED_TAGS = ["loss/g/mel", "loss/g/total"]

_TENSORBOARD = None
_TENSORBOARD_URL = None
_TENSORBOARD_LOCK = threading.Lock()


def _pinned_url(url: str, tags) -> str:
    """Ссылка на TensorBoard с закреплёнными карточками нужных скаляров."""
    cards = [{"plugin": "scalars", "tag": tag} for tag in tags]
    query = urllib.parse.quote(json.dumps(cards, separators=(",", ":")))
    return f"{url.rstrip('/')}/?pinnedCards={query}"


def launch_tensorboard(logs_dir: str, port: int = DEFAULT_TENSORBOARD_PORT) -> str:
    """Запускает TensorBoard в фоновом потоке.

    Возвращает ссылку с закреплёнными карточками или строку 'Ошибка...'.
    Сервер живёт столько же, сколько процесс интерфейса.
    """
    global _TENSORBOARD, _TENSORBOARD_URL

    os.makedirs(logs_dir, exist_ok=True)

    with _TENSORBOARD_LOCK:
        if _TENSORBOARD_URL:
            return _TENSORBOARD_URL
        try:
            from tensorboard import program

            board = program.TensorBoard()
            board.configure(
                argv=[None, "--logdir", logs_dir, "--port", str(port), "--path_prefix", "/tensorboard"]
            )
            url = board.launch()
        except Exception as error:
            return f"Ошибка запуска TensorBoard: {error}"
        _TENSORBOARD = board
        _TENSORBOARD_URL = _pinned_url(url, PINNED_TAGS)
        return _TENSORBOARD_URL
