import os
import re

import requests
from tqdm import tqdm

from assets.pretrains import BASE_URL as PRETRAINS_BASE_URL
from assets.pretrains import NO_PRETRAIN, has_pretrains, pretrain_files, pretrain_rates, pretrain_sets, rate_label

PREDICTORS = "https://huggingface.co/Politrees/RVC_resources/resolve/main/predictors/"
EMBEDDERS = "https://huggingface.co/Politrees/RVC_resources/resolve/main/embedders/pytorch/"
FLASH_SR = "https://huggingface.co/datasets/jakeoneijk/FlashSR_weights/resolve/main/"

# Корень репозитория, а не текущий каталог: обучение идёт в подпроцессах с cwd = корень.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PREDICTORS_DIR = os.path.join(PROJECT_ROOT, "assets", "models", "predictors")
EMBEDDERS_DIR = os.path.join(PROJECT_ROOT, "assets", "models", "embedders")
FLASH_SR_DIR = os.path.join(PROJECT_ROOT, "assets", "models", "FlashSR")
PRETRAINS_DIR = os.path.join(PROJECT_ROOT, "assets", "models", "pretrains")

# Создаем папки, если их нет
os.makedirs(PREDICTORS_DIR, exist_ok=True)
os.makedirs(EMBEDDERS_DIR, exist_ok=True)
os.makedirs(FLASH_SR_DIR, exist_ok=True)
os.makedirs(PRETRAINS_DIR, exist_ok=True)


def dl_model(link, model_name, dir_name):
    file_path = os.path.join(dir_name, model_name)
    if os.path.exists(file_path):
        return  # Пропускаем загрузку, если файл уже существует

    r = requests.get(f"{link}{model_name}", stream=True)
    r.raise_for_status()

    total_size = int(r.headers.get("content-length", 0))
    with (
        open(file_path, "wb") as f,
        tqdm(
            desc=f"Установка {model_name}",
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar,
    ):
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


def check_and_install_models(include_flashsr=False):
    try:
        for model in ["rmvpe.pt", "hpa-rmvpe.pt"]:
            dl_model(PREDICTORS, model, PREDICTORS_DIR)

        for model in [
            # "hubert_base.pt",
            "contentvec_base.pt",
            # "korean_hubert_base.pt",
            # "chinese_hubert_base.pt",
            # "japanese_hubert_base.pt",
            # "portuguese_hubert_base.pt"
        ]:
            dl_model(EMBEDDERS, model, EMBEDDERS_DIR)

        if include_flashsr:
            for model in ["sr_vocoder.pth", "student_ldm.pth", "vae.pth"]:
                dl_model(FLASH_SR, model, FLASH_SR_DIR)

    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")


def download_with_progress(url, path):
    """Качает файл, отдавая строки прогресса. Качает во временный файл."""
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise ValueError(f"Не удалось скачать {url}: HTTP {response.status_code}")

    total = int(response.headers.get("content-length", 0))
    done, last_shown = 0, -1
    tmp_path = path + ".part"
    with open(tmp_path, "wb") as out:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            out.write(chunk)
            done += len(chunk)
            if total:
                percent = done * 100 // total
                if percent >= last_shown + 5:
                    last_shown = percent
                    yield f"  ⬇ {os.path.basename(path)}: {percent}% ({done // 1024 // 1024}/{total // 1024 // 1024} МБ)"
    os.replace(tmp_path, path)
    yield f"  ✓ {os.path.basename(path)} ({done // 1024 // 1024} МБ)"


def ensure_pretrains(vocoder, choice, sample_rate):
    """Скачивает претрейн вокодера при необходимости.

    Генератор: отдаёт строки для журнала, возвращает (путь G, путь D).
    Вызывать через `g, d = yield from ensure_pretrains(...)`.
    """
    rate = rate_label(sample_rate)
    if not has_pretrains(vocoder):
        raise ValueError(f"У вокодера «{vocoder}» нет встроенных претрейнов — выберите «{NO_PRETRAIN}» или укажите свои G/D.")
    if choice not in pretrain_sets(vocoder):
        raise ValueError(f"Претрейн «{choice}» несовместим с вокодером «{vocoder}»: наборы привязаны к архитектуре генератора.")
    files = pretrain_files(vocoder, choice, sample_rate)
    if files is None:
        rates = ", ".join(pretrain_rates(vocoder, choice))
        raise ValueError(f"Претрейн «{choice}» доступен только для {rates}.")

    d_file, g_file = files
    safe_name = re.sub(r"[^\w\-.]", "_", choice)
    paths = {}
    for kind, remote_file in (("D", d_file), ("G", g_file)):
        local_path = os.path.join(PRETRAINS_DIR, f"{safe_name}_{rate}_{kind}.pth")
        if os.path.isfile(local_path):
            yield f"  ✓ {os.path.basename(local_path)} уже установлен"
        else:
            remote = f"{PRETRAINS_BASE_URL}{rate}/{remote_file}"
            yield f"⬇ Скачиваю {choice} ({rate}, {kind})…"
            yield from download_with_progress(remote, local_path)
        paths[kind] = local_path
    return paths["G"], paths["D"]


if __name__ == "__main__":
    check_and_install_models()
