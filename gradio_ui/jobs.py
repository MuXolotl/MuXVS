"""Запуск долгих задач (шаги обучения) с живым журналом и кнопкой остановки.

Скрипты пайплайна читают аргументы из `sys.argv` на уровне модуля,
поэтому запускаются отдельным процессом — как в официальном RVC WebUI.
"""

import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque

import gradio as gr

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAIL_LINES = 200  # Сколько последних строк журнала показывать

_lock = threading.Lock()
_process = None
_stop_requested = False


def busy() -> bool:
    with _lock:
        return _process is not None and _process.poll() is None


def command(*args) -> list:
    """Команда запуска модуля проекта тем же интерпретатором, что и интерфейс."""
    return [sys.executable or "python", "-m", *[str(arg) for arg in args]]


def request_stop() -> str:
    """Обработчик кнопки «Остановить». Вызывается с queue=False."""
    global _process, _stop_requested
    with _lock:
        _stop_requested = True
        proc = _process
    if proc is not None and proc.poll() is None:
        try:
            if os.name == "nt":
                proc.terminate()
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass
    return "Останавливаю…"


def _spawn(cmd: list) -> subprocess.Popen:
    global _process
    kwargs = {
        "cwd": PROJECT_ROOT,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "bufsize": 1,
        "env": {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    with _lock:
        _process = subprocess.Popen(cmd, **kwargs)  # noqa: S603 — команда собрана из констант
        return _process


def run_job(title: str, commands: list):
    """Генератор для кнопки запуска: выполняет команды по очереди, отдаёт журнал."""
    global _process, _stop_requested
    if busy():
        raise gr.Error("Другая задача ещё выполняется — дождитесь её или остановите.")

    if commands and isinstance(commands[0], str):
        commands = [commands]
    with _lock:
        _stop_requested = False

    lines = deque(maxlen=TAIL_LINES)
    lines.append(f"▶ {title}")
    yield "\n".join(lines)

    try:
        for cmd in commands:
            with _lock:
                if _stop_requested:
                    break
            lines.append(f"$ {' '.join(cmd)}")
            proc = _spawn(cmd)
            last_yield = 0.0
            for raw in proc.stdout:
                text = raw.replace("\r", "\n").strip()
                if text:
                    lines.extend(part for part in text.split("\n") if part.strip())
                if time.monotonic() - last_yield >= 0.3:
                    last_yield = time.monotonic()
                    yield "\n".join(lines)
            proc.wait()
            with _lock:
                stopped = _stop_requested
            if stopped:
                lines.append("⏹ Остановлено пользователем.")
                yield "\n".join(lines)
                return
            if proc.returncode != 0:
                lines.append(f"❌ Ошибка (код {proc.returncode}). Смотрите журнал выше.")
                yield "\n".join(lines)
                return
    finally:
        with _lock:
            _process = None
            _stop_requested = False

    lines.append("✓ Готово.")
    yield "\n".join(lines)
