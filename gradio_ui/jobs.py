"""Запуск долгих задач (шаги обучения) с живым журналом и кнопкой остановки.

Скрипты пайплайна читают аргументы из `sys.argv` на уровне модуля,
поэтому запускаются отдельным процессом — как в официальном RVC WebUI.
"""

import io
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


def _kill_process_tree(proc: subprocess.Popen, sig: int) -> None:
    """Сигнал всей группе процесса (posix) или самому процессу (Windows)."""
    try:
        if os.name == "nt":
            proc.send_signal(sig)
        else:
            os.killpg(os.getpgid(proc.pid), sig)
    except (OSError, ProcessLookupError):
        # Процесс уже мёртв или группа недоступна — останавливать нечего.
        pass


def request_stop() -> str:
    """Обработчик кнопки «Остановить». Вызывается с queue=False."""
    global _process, _stop_requested
    with _lock:
        _stop_requested = True
        proc = _process
    if proc is None or proc.poll() is not None:
        return "Нет запущенного процесса."
    _kill_process_tree(proc, signal.SIGTERM)
    try:
        proc.wait(timeout=5)
        return "Процесс остановлен."
    except subprocess.TimeoutExpired:
        pass
    # Процесс пережил SIGTERM (зависший ввод-вывод, игнор сигнала) — добиваем.
    _kill_process_tree(proc, signal.SIGKILL if os.name != "nt" else signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        return "Не удалось остановить процесс — убейте его вручную."
    return "Процесс остановлен."


def _spawn(cmd: list) -> subprocess.Popen:
    global _process
    kwargs = {
        "cwd": PROJECT_ROOT,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
        "env": {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    with _lock:
        _process = subprocess.Popen(cmd, **kwargs)  # noqa: S603 — команда собрана из констант
        # Текстовая обёртка без трансляции \r: это живой прогресс tqdm, а не конец строки.
        _process.stdout = io.TextIOWrapper(_process.stdout, encoding="utf-8", errors="replace", newline="")
        return _process


def run_job(title: str, commands: list, prefix: str = ""):
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
    if prefix:
        lines.extend(prefix.split("\n"))
    yield "\n".join(lines)

    try:
        for cmd in commands:
            with _lock:
                if _stop_requested:
                    break
            lines.append(f"$ {' '.join(cmd)}")
            proc = _spawn(cmd)
            last_yield = 0.0
            buffer = ""
            live = ""
            while True:
                # Читаем посимвольно: tqdm минутами шлёт только \r без \n,
                # построчное чтение на это время глухо виснет.
                chunk = proc.stdout.read(1)
                if chunk:
                    if chunk in "\r\n":
                        text = buffer.strip()
                        buffer = ""
                        if chunk == "\n":
                            live = ""
                            if text:
                                lines.append(text)
                        elif text:
                            live = text
                    else:
                        buffer += chunk
                elif proc.poll() is not None:
                    break
                else:
                    time.sleep(0.05)
                if time.monotonic() - last_yield >= 0.3:
                    last_yield = time.monotonic()
                    yield "\n".join([*lines, live] if live else lines)
            tail = buffer.strip()
            if tail:
                lines.append(tail)
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
        # Флаг не сбрасываем: следующий запуск выставит его сам, читателей между запусками нет.
        with _lock:
            _process = None

    lines.append("✓ Готово.")
    yield "\n".join(lines)
