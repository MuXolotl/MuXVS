"""Вкладка «Инструменты»: проверка моделей, разделение чекпоинта, апскейл аудио."""

import os

import gradio as gr

UPSCALE_OUTPUT_DIR = os.path.join(os.getcwd(), "output", "Upscale_output")
DEFAULT_SPLIT_DIR = os.path.join(os.getcwd(), "assets", "models", "pretrains")

MODEL_INFO_KEYS = [
    ("model_name", "Модель"),
    ("version", "Версия"),
    ("vocoder", "Вокодер"),
    ("sr", "Частота"),
    ("epoch", "Эпоха"),
    ("step", "Шаг"),
    ("f0", "С питчем"),
]


def _uploaded_path(item):
    if item is None:
        return None
    return getattr(item, "name", None) or (item if isinstance(item, str) else None)


def _inspect_model(model_file) -> str:
    path = _uploaded_path(model_file)
    if not path or not os.path.isfile(path):
        raise gr.Error("Прикрепите .pth файл модели.")
    try:
        import torch
    except ImportError as error:
        raise gr.Error("PyTorch не установлен.") from error
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise gr.Error(f"Не удалось прочитать файл: {error}") from error
    if not isinstance(checkpoint, dict):
        raise gr.Error("Файл не похож на модель.")

    if "generator" in checkpoint and "discriminator" in checkpoint:
        generator = checkpoint.get("generator") or {}
        return "\n".join(
            [
                "Единый чекпоинт обучения (G + D), не модель для инференса.",
                f"Эпоха: {checkpoint.get('epoch', '—')}",
                f"Learning rate: {generator.get('lr', checkpoint.get('learning_rate', '—'))}",
                "",
                "Разделите его ниже, чтобы получить претрейны G/D.",
            ],
        )

    lines = [f"Файл: {os.path.basename(path)} ({os.path.getsize(path) / 1024 / 1024:.1f} МБ)"]
    lines.extend(f"{label}: {checkpoint[key]}" for key, label in MODEL_INFO_KEYS if key in checkpoint)
    weights = checkpoint.get("weight")
    if isinstance(weights, dict):
        lines.append(f"Слоёв: {len(weights)}")
    if checkpoint.get("version") != "v2" or not checkpoint.get("f0", 1):
        lines.append("")
        lines.append("⚠️ MuXVS поддерживает только v2-модели с питчем.")
    return "\n".join(lines)


def _split_checkpoint(checkpoint_file, output_dir) -> str:
    from rvc.utils.split_checkpoint import split_checkpoint

    path = _uploaded_path(checkpoint_file)
    if not path:
        raise gr.Error("Прикрепите checkpoint.pth.")
    try:
        return split_checkpoint(path, output_dir.strip() or DEFAULT_SPLIT_DIR)
    except (FileNotFoundError, ValueError) as error:
        raise gr.Error(str(error)) from error


def _upscale_audio(audio_file, overlap, progress=gr.Progress(track_tqdm=True)) -> gr.Audio:
    path = _uploaded_path(audio_file)
    if not path:
        raise gr.Error("Прикрепите аудиофайл.")
    try:
        from rvc._library.config import Config
        from rvc.inference.modules.audio_upscaler import upscale
    except ImportError as error:
        raise gr.Error(f"Нет зависимостей апскейла: {error}") from error

    os.makedirs(UPSCALE_OUTPUT_DIR, exist_ok=True)
    progress(0.1, desc="Улучшаем качество…")
    try:
        upscale(path, UPSCALE_OUTPUT_DIR, int(overlap), Config().device)
    except Exception as error:
        raise gr.Error(f"Апскейл не удался: {error}") from error
    output_path = os.path.join(UPSCALE_OUTPUT_DIR, os.path.basename(path))
    if not os.path.isfile(output_path):
        raise gr.Error(f"Результат не найден: {output_path}")
    return gr.Audio(output_path, label=os.path.basename(output_path))


def tools_tab():
    with gr.Accordion("Проверка модели (.pth)", open=True):
        model_file = gr.File(label="Файл модели", file_types=[".pth"], file_count="single")
        inspect_btn = gr.Button("Проверить", variant="primary")
        model_info = gr.Textbox(label="Сведения", lines=10, interactive=False)
        inspect_btn.click(_inspect_model, inputs=model_file, outputs=model_info)

    with gr.Accordion("Разделение чекпоинта (G / D)", open=False):
        gr.Markdown("Единый `checkpoint.pth` → `G_pretrain.pth` + `D_pretrain.pth`.")
        with gr.Row():
            checkpoint_file = gr.File(label="Чекпоинт", file_types=[".pth"], file_count="single", scale=2)
            split_dir = gr.Textbox(DEFAULT_SPLIT_DIR, label="Папка результата", scale=3)
        split_btn = gr.Button("Разделить", variant="primary")
        split_message = gr.Textbox(label="Результат", lines=4, interactive=False)
        split_btn.click(_split_checkpoint, inputs=[checkpoint_file, split_dir], outputs=split_message)

    with gr.Accordion("Апскейл аудио (FlashSR)", open=False):
        with gr.Row():
            upscale_in = gr.Audio(label="Исходное аудио", type="filepath")
            upscale_out = gr.Audio(label="Улучшенное аудио", interactive=False)
        with gr.Row():
            overlap = gr.Slider(minimum=1, maximum=8, step=1, value=2, label="Перекрытие")
            upscale_btn = gr.Button("Улучшить", variant="primary")
        upscale_btn.click(_upscale_audio, inputs=[upscale_in, overlap], outputs=upscale_out)
