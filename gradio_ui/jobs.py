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
_process_group_id = None
_stop_requested = False


def busy() -> bool:
    with _lock:
        proc = _process
        group_id = _process_group_id
    if proc is None:
        return False
    if proc.poll() is None:
        return True
    if os.name != "nt" and group_id is not None:
        try:
            os.killpg(group_id, 0)
            return True
        except (OSError, ProcessLookupError):
            pass
    return False


def command(*args) -> list:
    """Команда запуска модуля проекта тем же интерпретатором, что и интерфейс."""
    return [sys.executable or "python", "-m", *[str(arg) for arg in args]]


def _kill_process_tree(proc: subprocess.Popen, sig: int, process_group_id=None) -> None:
    """Останавливает процесс и его потомков, даже если лидер уже завершился."""
    if os.name == "nt":
        # Popen.terminate() завершает только родителя; taskkill /T нужен,
        # чтобы вместе с ним остановить multiprocessing/DataLoader workers.
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return
    try:
        # Не вызывать os.getpgid(proc.pid) здесь: после выхода лидера это
        # невозможно, хотя его дочерние процессы ещё могут оставаться живы.
        os.killpg(process_group_id if process_group_id is not None else proc.pid, sig)
    except (OSError, ProcessLookupError):
        pass


def request_stop() -> str:
    """Обработчик кнопки «Остановить». Вызывается с queue=False."""
    global _process, _stop_requested
    with _lock:
        _stop_requested = True
        proc = _process
        process_group_id = _process_group_id
    if proc is None:
        return "Нет запущенного процесса."

    _kill_process_tree(proc, signal.SIGTERM, process_group_id)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass

    # Не считаем задачу остановленной лишь потому, что вышел родительский
    # Python-процесс: воркеры DataLoader могут пережить его и держать ОЗУ.
    _kill_process_tree(proc, signal.SIGKILL if os.name != "nt" else signal.SIGTERM, process_group_id)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        return "Не удалось остановить процесс — проверьте оставшиеся дочерние процессы."
    return "Процесс и его дочерние процессы остановлены."


def _release_inference_cache() -> None:
    """Освобождает кэш инференса: обучающему процессу нужна вся память GPU.

    Импорт ленивый — `rvc.inference.infer` тянет torch, а интерфейс запускается без него.
    """
    try:
        from rvc.inference.infer import release_pipeline
    except Exception:  # noqa: BLE001 — без torch/зависимостей выгружать нечего
        return
    released = release_pipeline()
    if released:
        print(f"[i] Модель '{released}' выгружена из памяти перед запуском задачи.", flush=True)


def _spawn(cmd: list) -> subprocess.Popen:
    global _process, _process_group_id
    _release_inference_cache()
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
        _process_group_id = _process.pid if os.name != "nt" else None
        # Текстовая обёртка без трансляции \r: это живой прогресс tqdm, а не конец строки.
        _process.stdout = io.TextIOWrapper(_process.stdout, encoding="utf-8", errors="replace", newline="")
        return _process


def run_job(title: str, commands: list, prefix: str = ""):
    """Генератор для кнопки запуска: выполняет команды по очереди, отдаёт журнал."""
    global _process, _process_group_id, _stop_requested
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
                            # Jupyter/Colab не перерисовывает \r в консольной выдаче:
                            # там печатаем завершённый tqdm-бар один раз, по LF.
                            # В обычном TTY построчно завершаем уже обновляемый бар.
                            console_text = live or text
                            if console_text:
                                lines.append(console_text)
                                if live and sys.stdout.isatty():
                                    print("", flush=True)
                                else:
                                    print(console_text, flush=True)
                            live = ""
                        elif chunk == "\r":
                            if text:
                                live = text
                                if sys.stdout.isatty():
                                    print("\r" + live, end="", flush=True)
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
                print(tail, flush=True)
                lines.append(tail)
            if live:
                if sys.stdout.isatty():
                    print("", flush=True)
                else:
                    print(live, flush=True)
                lines.append(live)
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
            _process_group_id = None

    lines.append("✓ Готово.")
    yield "\n".join(lines)
