import os
import shutil
import subprocess
from functools import lru_cache

import numpy as np

FFMPEG = "ffmpeg"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SOXR_RESAMPLER = "resampler=soxr:precision=28"
SWR_RESAMPLER = "filter_size=128"

LOAD_TIMEOUT = 600  # секунд на чтение файла
SAVE_TIMEOUT = 300  # секунд на запись файла


@lru_cache(maxsize=1)
def ffmpeg_bin() -> str:
    """Возвращает путь к FFmpeg: сначала PATH, затем бинарь из корня."""
    found = shutil.which(FFMPEG)
    if found:
        return found

    bundled = os.path.join(PROJECT_ROOT, FFMPEG + (".exe" if os.name == "nt" else ""))
    return bundled if os.path.isfile(bundled) else FFMPEG


@lru_cache(maxsize=1)
def resampler_options() -> str:
    """Определяет, каким ресемплером умеет работать установленный FFmpeg.

    Проба выполняется один раз на процесс: если сборка собрана без libsoxr,
    остаёмся на встроенном swr с увеличенным фильтром, а не падаем на каждом файле.
    """
    probe = [
        ffmpeg_bin(),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=channel_layout=mono:sample_rate=8000",
        "-t",
        "0.01",
        "-af",
        f"aresample=16000:{SOXR_RESAMPLER}",
        "-f",
        "null",
        "-",
    ]
    try:
        result = subprocess.run(probe, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    except OSError:
        return SWR_RESAMPLER

    return SOXR_RESAMPLER if result.returncode == 0 else SWR_RESAMPLER


def _ffmpeg_not_found() -> RuntimeError:
    """Одна и та же подсказка для чтения и записи: без FFmpeg не работает ни то ни другое."""
    return RuntimeError(f"FFmpeg не найден: установите его или положите '{FFMPEG}' в каталог проекта.")


def _ffmpeg_failure(process: subprocess.CompletedProcess, file: str) -> str:
    """Собирает сообщение об ошибке из stderr FFmpeg — без него причина отказа не видна."""
    lines = [line.strip() for line in process.stderr.decode(errors="replace").splitlines() if line.strip()]
    return f"FFmpeg не смог обработать '{file}': {lines[-1] if lines else f'код возврата {process.returncode}'}"


def load_audio(file: str, sample_rate: int) -> np.ndarray:
    """Читает аудио любым кодеком, который понимает FFmpeg, и приводит его к моно с частотой `sample_rate`."""
    # Из UI/CLI путь может приехать в кавычках и с переводами строк — срезаем и то и другое
    file = file.strip().strip("\"'").strip()
    cmd = [
        ffmpeg_bin(),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-threads",
        "0",
        "-i",
        file,
        "-af",
        f"aresample={int(sample_rate)}:{resampler_options()}",
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(int(sample_rate)),
        "-",
    ]

    try:
        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=LOAD_TIMEOUT, check=False)
    except FileNotFoundError as error:
        raise _ffmpeg_not_found() from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"FFmpeg не уложился в {LOAD_TIMEOUT} с при чтении '{file}'.") from error

    if process.returncode != 0:
        raise RuntimeError(_ffmpeg_failure(process, file))
    if not process.stdout:
        raise RuntimeError(f"В файле '{file}' нет аудиоданных.")

    return np.frombuffer(process.stdout, np.float32).flatten()


def save_audio(
    audio_data: np.ndarray,
    sample_rate: int,
    output_path: str,
    output_format: str = "wav",
    stereo: bool = False,
) -> str:
    """Сохраняет аудио используя прямой вызов FFmpeg через pipe."""
    # Конвертируем в int16 или float32 в зависимости от формата
    if output_format in ["wav", "flac"]:
        # Для lossless форматов используем 24-bit
        audio_data = np.clip(audio_data, -1.0, 1.0)
        # Конвертируем в 32-bit float для максимальной точности
        audio_bytes = audio_data.astype(np.float32).tobytes()
        input_format = "f32le"
    else:
        # Для lossy форматов используем 16-bit; клиппинг обязателен
        audio_data = np.clip(audio_data, -1.0, 1.0)
        audio_int16 = (audio_data * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        input_format = "s16le"

    channels = 2 if stereo else 1

    # Базовые параметры FFmpeg
    cmd = [
        ffmpeg_bin(),
        "-y",  # Перезаписывать выходной файл
        "-f",
        input_format,  # Формат входных данных
        "-ar",
        str(sample_rate),  # Частота дискретизации
        "-ac",
        "1",  # Входные каналы (всегда моно из RVC)
        "-i",
        "pipe:0",  # Читать из stdin
        "-ac",
        str(channels),  # Выходные каналы
    ]

    # Настройки качества для каждого формата
    format_settings = {
        "wav": [
            "-c:a",
            "pcm_f32le",  # 32-bit float PCM для максимального качества
            "-sample_fmt",
            "flt",
        ],
        "flac": [
            "-c:a",
            "flac",
            "-compression_level",
            "12",  # Максимальное сжатие (без потерь)
            "-sample_fmt",
            "s32",  # 32-bit для максимального качества
        ],
        "mp3": [
            "-c:a",
            "libmp3lame",
            "-b:a",
            "320k",  # Максимальный битрейт
            "-q:a",
            "0",  # Наилучшее качество
        ],
        "ogg": [
            "-c:a",
            "libvorbis",
            "-q:a",
            "10",  # Максимальное качество (500kbps+)
        ],
        "m4a": [
            "-c:a",
            "aac",
            "-b:a",
            "320k",  # Максимальный битрейт
            "-q:a",
            "2",  # Максимальное качество
            "-aac_coder",
            "twoloop",  # Лучший кодировщик
            "-profile:a",
            "aac_low",
        ],
    }

    if output_format in format_settings:
        cmd.extend(format_settings[output_format])
    else:
        raise ValueError(f"Неподдерживаемый формат: {output_format}")

    # Добавляем выходной файл
    cmd.append(output_path)

    # Запускаем FFmpeg
    try:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as error:
        raise _ffmpeg_not_found() from error

    try:
        stderr = process.communicate(input=audio_bytes, timeout=SAVE_TIMEOUT)[1]
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        raise RuntimeError(f"FFmpeg не уложился в {SAVE_TIMEOUT} с при записи '{output_path}'") from None

    if process.returncode != 0:
        detail = stderr.decode(errors="replace").strip().splitlines()
        raise RuntimeError(f"FFmpeg не смог записать '{output_path}': {detail[-1] if detail else process.returncode}")

    return output_path
