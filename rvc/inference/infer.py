import asyncio
import gc
import os

import edge_tts
import gradio as gr
import torch

from rvc._library.algorithm.synthesizers import Synthesizer
from rvc._library.audio import load_audio, save_audio
from rvc._library.config import Config
from rvc._library.embedders.fairseq import load_model
from rvc.inference.modules.audio_upscaler import upscale
from rvc.inference.pipeline import VC

# Определяем пути к папкам и файлам (константы)
RVC_MODELS_DIR = os.path.join(os.getcwd(), "models", "RVC_models")
OUTPUT_DIR = os.path.join(os.getcwd(), "output", "RVC_output")
EMBEDDER_PATH = os.path.join(os.getcwd(), "assets", "models", "embedders", "contentvec_base.pt")

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".opus", ".wma", ".aiff"}

# Создаем папки, если их нет
os.makedirs(RVC_MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Инициализация конфигурации
config = Config()


# Отображает прогресс выполнения задачи.
def display_progress(percent, message, is_print, progress=gr.Progress()):
    if is_print:
        print(message)
    progress(percent, desc=message)


# Загружает модель RVC и индекс по имени модели.
def load_rvc_model(rvc_model):
    model_dir = os.path.join(RVC_MODELS_DIR, rvc_model)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Папка модели {rvc_model} не найдена в {RVC_MODELS_DIR}")

    # Находим файл модели с расширением .pth
    rvc_model_path = next((os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".pth")), None)
    # Находим файл индекса с расширением .index
    rvc_index_path = next((os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".index")), None)

    # Проверяем, существует ли файл модели
    if not rvc_model_path:
        raise FileNotFoundError(f"Модель {rvc_model} не содержит .pth файла!")

    return rvc_model_path, rvc_index_path


# Загружает семантический эмбеддер
def load_embedder(model_path):
    embedder = load_model(model_path).to(config.device).eval()
    return embedder


# Получает конвертер голоса
def get_vc(model_path):
    # Загружаем состояние модели
    cpt = torch.load(model_path, map_location="cpu", weights_only=True)
    if "config" not in cpt or "weight" not in cpt:
        raise ValueError(f"Некорректный формат модели {model_path}. Используйте модель RVC.")

    # MuXVS поддерживает только v2-модели с питчем
    model_name = os.path.basename(model_path)
    if cpt.get("version") != "v2":
        raise ValueError(
            f"Модель '{model_name}' не v2 (version={cpt.get('version')!r}). MuXVS поддерживает только v2-модели.",
        )
    if not cpt.get("f0", 1):
        raise ValueError(
            f"Модель '{model_name}' обучена без питча (f0=False). MuXVS поддерживает только v2-модели с питчем.",
        )

    # Извлекаем параметры модели
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    vocoder = cpt.get("vocoder", "HiFi-GAN")

    # Инициализируем синтезатор
    net_g = Synthesizer(*cpt["config"], text_enc_hidden_dim=768, vocoder=vocoder)

    # Удаляем ненужный слой
    del net_g.enc_q
    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g = net_g.to(config.device).float().eval()

    # Инициализируем объект конвертера голоса
    vc = VC(tgt_sr, config)
    return cpt, net_g, tgt_sr, vc


# Синтезирует текст в речь с использованием edge_tts.
async def text_to_speech(voice, text, rate, volume, pitch, output_path):
    if not -100 <= rate <= 100 or not -100 <= volume <= 100 or not -100 <= pitch <= 100:
        raise ValueError("Параметры Rate, Volume и Pitch должны быть в диапазоне от -100 до +100.")

    communicate = edge_tts.Communicate(voice=voice, text=text, rate=f"{rate:+d}%", volume=f"{volume:+d}%", pitch=f"{pitch:+d}Hz")
    await communicate.save(output_path)


# Собирает путь выходного файла вида `<имя>_(<модель>).<формат>`
def build_output_path(input_path, output_dir, rvc_model, output_format):
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    if len(base_name) > 100:  # Сменить имя выходного файла, если длина исходного более 100 символов
        gr.Warning("Имя файла превышает 100 символов и будет сокращено для удобства.")
        base_name = f"{base_name[:25]}... (Made_in_MuXVS)"
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"{base_name}_({rvc_model}).{output_format}")


# Загружает эмбеддер, модель и конвертер голоса одним набором
def load_pipeline(rvc_model):
    display_progress(0.1, "Загружаем семантический эмбеддер...", False)
    embedder_model = load_embedder(EMBEDDER_PATH)
    display_progress(0.2, f"Загружаем модель '{rvc_model}'...", False)
    model_path, index_path = load_rvc_model(rvc_model)
    display_progress(0.3, "Получаем конвертер голоса...", False)
    cpt, net_g, tgt_sr, vc = get_vc(model_path)
    return {
        "embedder": embedder_model,
        "cpt": cpt,
        "net_g": net_g,
        "tgt_sr": tgt_sr,
        "vc": vc,
        "index_path": index_path,
    }


# Выгружает пайплайн из памяти
def free_pipeline(pipe):
    pipe.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Конвертирует один файл уже загруженным пайплайном
def convert_one(
    pipe,
    input_path,
    output_path,
    f0_method="rmvpe",
    f0_min=50,
    f0_max=1100,
    rvc_pitch=0,
    protect=0.5,
    index_rate=0.25,
    volume_envelope=1.0,
    autopitch=False,
    autopitch_threshold=155.0,
    autotune=False,
    autotune_tonic="C",
    autotune_scale="chromatic",
    autotune_strength=1.0,
    stereo_sound=False,
    output_format="wav",
):
    display_progress(0.4, "Загружаем аудио...", False)
    audio = load_audio(input_path, 16000)

    display_progress(0.5, f"[🌌] Преобразуем аудио '{os.path.basename(input_path)}'...", True)
    audio_opt = pipe["vc"].pipeline(
        model=pipe["embedder"],
        net_g=pipe["net_g"],
        sid=0,
        audio=audio,
        pitch=0 if autopitch else rvc_pitch,
        f0_min=f0_min,
        f0_max=f0_max,
        f0_method=f0_method,
        file_index=pipe["index_path"],
        index_rate=index_rate,
        volume_envelope=volume_envelope,
        protect=protect,
        autopitch=autopitch,
        autopitch_threshold=autopitch_threshold,
        autotune=autotune,
        autotune_tonic=autotune_tonic,
        autotune_scale=autotune_scale,
        autotune_strength=autotune_strength,
    )

    display_progress(0.8, "[💫] Сохраняем результат...", True)
    save_audio(audio_opt, pipe["tgt_sr"], output_path, output_format, stereo_sound)
    return output_path


# Выполнение инференса с использованием RVC
def rvc_infer(
    rvc_model=None,
    input_path=None,
    f0_method="rmvpe",
    f0_min=50,
    f0_max=1100,
    rvc_pitch=0,
    protect=0.5,
    index_rate=0.25,
    volume_envelope=1.0,
    autopitch=False,
    autopitch_threshold=155.0,
    autotune=False,
    autotune_tonic="C",
    autotune_scale="chromatic",
    autotune_strength=1.0,
    audio_upscaling=False,  # FlashSR
    stereo_sound=False,
    output_format="wav",
    progress=gr.Progress(track_tqdm=True),
):
    if not rvc_model:
        raise ValueError("Не выбрана модель для RVC-инференса")
    if not input_path or not os.path.exists(input_path):
        raise FileNotFoundError(f"Файл '{input_path}' не найден!")

    display_progress(0, "\n[⚙️] Запуск конвейера генерации...", True)
    pipe = load_pipeline(rvc_model)
    try:
        output_path = build_output_path(input_path, OUTPUT_DIR, rvc_model, output_format)
        convert_one(
            pipe,
            input_path,
            output_path,
            f0_method=f0_method,
            f0_min=f0_min,
            f0_max=f0_max,
            rvc_pitch=rvc_pitch,
            protect=protect,
            index_rate=index_rate,
            volume_envelope=volume_envelope,
            autopitch=autopitch,
            autopitch_threshold=autopitch_threshold,
            autotune=autotune,
            autotune_tonic=autotune_tonic,
            autotune_scale=autotune_scale,
            autotune_strength=autotune_strength,
            stereo_sound=stereo_sound,
            output_format=output_format,
        )

        if audio_upscaling:
            display_progress(0.9, "[🚀] Улучшаем качество аудио...", True)
            upscale(output_path, OUTPUT_DIR, 2, config.device)
    finally:
        display_progress(0.95, "Освобождаем память...", False)
        free_pipeline(pipe)

    display_progress(1.0, f"[✅] Преобразование завершено — {output_path}", True)
    return gr.Audio(output_path, label=os.path.basename(output_path))


# Собирает список аудиофайлов: папка и/или загруженные файлы
def collect_batch_inputs(dir_input, files):
    inputs = []
    if dir_input and str(dir_input).strip():
        folder = str(dir_input).strip()
        if not os.path.isdir(folder):
            raise ValueError(f"Папка не найдена: {folder}")
        for name in sorted(os.listdir(folder)):
            path = os.path.join(folder, name)
            if os.path.isfile(path) and os.path.splitext(name)[1].lower() in AUDIO_EXTENSIONS:
                inputs.append(path)
    for item in files or []:
        path = getattr(item, "name", None) or (item if isinstance(item, str) else None)
        if path and os.path.isfile(path):
            inputs.append(path)
    # Убираем дубликаты, сохраняя порядок
    return list(dict.fromkeys(inputs))


# Пакетная конвертация: модель грузится один раз, файлы идут по очереди
def rvc_batch_infer(
    rvc_model=None,
    dir_input=None,
    files=None,
    output_dir=None,
    f0_method="rmvpe",
    f0_min=50,
    f0_max=1100,
    rvc_pitch=0,
    protect=0.5,
    index_rate=0.25,
    volume_envelope=1.0,
    autopitch=False,
    autopitch_threshold=155.0,
    autotune=False,
    autotune_tonic="C",
    autotune_scale="chromatic",
    autotune_strength=1.0,
    audio_upscaling=False,
    stereo_sound=False,
    output_format="wav",
    progress=gr.Progress(track_tqdm=True),
):
    if not rvc_model:
        raise ValueError("Не выбрана модель для RVC-инференса")
    inputs = collect_batch_inputs(dir_input, files)
    if not inputs:
        raise ValueError("Нет аудиофайлов: укажите папку или прикрепите файлы.")

    target_dir = str(output_dir).strip() if output_dir and str(output_dir).strip() else OUTPUT_DIR
    os.makedirs(target_dir, exist_ok=True)

    print(f"\n[⚙️] Пакетная конвертация: {len(inputs)} файл(ов), модель '{rvc_model}'")
    pipe = load_pipeline(rvc_model)
    done, failed = [], []
    try:
        for num, input_path in enumerate(inputs, 1):
            progress(num / len(inputs), desc=f"[{num}/{len(inputs)}] {os.path.basename(input_path)}")
            try:
                output_path = build_output_path(input_path, target_dir, rvc_model, output_format)
                convert_one(
                    pipe,
                    input_path,
                    output_path,
                    f0_method=f0_method,
                    f0_min=f0_min,
                    f0_max=f0_max,
                    rvc_pitch=rvc_pitch,
                    protect=protect,
                    index_rate=index_rate,
                    volume_envelope=volume_envelope,
                    autopitch=autopitch,
                    autopitch_threshold=autopitch_threshold,
                    autotune=autotune,
                    autotune_tonic=autotune_tonic,
                    autotune_scale=autotune_scale,
                    autotune_strength=autotune_strength,
                    stereo_sound=stereo_sound,
                    output_format=output_format,
                )
                if audio_upscaling:
                    upscale(output_path, target_dir, 2, config.device)
                done.append(output_path)
            except Exception as error:
                failed.append(f"{os.path.basename(input_path)}: {error}")
    finally:
        free_pipeline(pipe)

    lines = [f"Готово: {len(done)}/{len(inputs)}. Папка: {target_dir}"]
    lines.extend(f"✓ {os.path.basename(path)}" for path in done)
    lines.extend(f"✗ {item}" for item in failed)
    print(f"[✅] Пакетная конвертация завершена — {len(done)}/{len(inputs)}")
    return "\n".join(lines)


def rvc_edgetts_infer(
    # RVC
    rvc_model=None,
    f0_method="rmvpe",
    f0_min=50,
    f0_max=1100,
    rvc_pitch=0,
    protect=0.5,
    index_rate=0.25,
    volume_envelope=1.0,
    autopitch=False,
    autopitch_threshold=155.0,
    autotune=False,
    autotune_tonic="C",
    autotune_scale="chromatic",
    autotune_strength=1.0,
    stereo_sound=False,
    output_format="wav",
    # EdgeTTS
    tts_voice=None,
    tts_text=None,
    tts_rate=0,
    tts_volume=0,
    tts_pitch=0,
    # FlashSR
    audio_upscaling=False,
    progress=gr.Progress(track_tqdm=True),
):
    if not tts_text:
        raise ValueError("Введите текст!")
    if not tts_voice:
        raise ValueError("Выберите голос!")

    display_progress(1.0, "[🎙️] Синтезируем речь...", False)
    input_path = os.path.join(OUTPUT_DIR, "TTS_Voice.wav")
    asyncio.run(text_to_speech(tts_voice, tts_text, tts_rate, tts_volume, tts_pitch, input_path))

    output_path = rvc_infer(
        rvc_model=rvc_model,
        input_path=input_path,
        f0_method=f0_method,
        f0_min=f0_min,
        f0_max=f0_max,
        rvc_pitch=rvc_pitch,
        protect=protect,
        index_rate=index_rate,
        volume_envelope=volume_envelope,
        autopitch=autopitch,
        autopitch_threshold=autopitch_threshold,
        autotune=autotune,
        autotune_tonic=autotune_tonic,
        autotune_scale=autotune_scale,
        autotune_strength=autotune_strength,
        audio_upscaling=audio_upscaling,
        stereo_sound=stereo_sound,
        output_format=output_format,
    )

    return input_path, output_path
