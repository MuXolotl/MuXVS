"""Вкладка «Обучение»: шаги как в официальном RVC WebUI.

1. Нарезка датасета → 2. Извлечение признаков → 3. Обучение (+ индекс).
Шаги запускаются отдельными процессами с общим живым журналом.
"""

import os
import shutil

import gradio as gr

from gradio_ui.jobs import command, request_stop, run_job

LOGS_DIR = os.path.join(os.getcwd(), "logs")
SAMPLE_RATES = [32000, 40000, 48000]
VOCODERS = ["HiFi-GAN", "MRF HiFi-GAN", "RefineGAN"]
OPTIMIZERS = ["AdamW", "AdaBelief", "PolOpt"]
F0_METHODS = ["rmvpe", "rmvpe+", "hpa-rmvpe"]
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac", ".wma", ".aiff", ".webm", ".mp4")


def _exp_dir(model_name: str) -> str:
    return os.path.join(LOGS_DIR, model_name.strip())


def _require_name(model_name: str) -> str:
    if not model_name or not model_name.strip():
        raise gr.Error("Укажите имя модели.")
    return model_name.strip()


def _input_root(model_name: str, files, folder: str) -> str:
    """Папка входа для нарезки: загруженные файлы копируются в датасет модели."""
    if files:
        dataset_dir = os.path.join(_exp_dir(model_name), "dataset")
        os.makedirs(dataset_dir, exist_ok=True)
        copied = 0
        for item in files:
            path = getattr(item, "name", None) or (item if isinstance(item, str) else None)
            if path and os.path.isfile(path) and path.lower().endswith(AUDIO_EXTENSIONS):
                shutil.copyfile(path, os.path.join(dataset_dir, os.path.basename(path)))
                copied += 1
        if not copied:
            raise gr.Error("Среди прикреплённых файлов нет аудио.")
        return dataset_dir
    if folder and folder.strip() and os.path.isdir(folder.strip()):
        return folder.strip()
    raise gr.Error("Прикрепите аудиофайлы или укажите папку с датасетом.")


def _slice_dataset(model_name, files, folder, sample_rate, segment_len, normalize):
    model_name = _require_name(model_name)
    input_root = _input_root(model_name, files, folder)
    yield from run_job(
        f"Нарезка датасета «{model_name}»",
        command(
            "rvc.training.preprocess.preprocess",
            _exp_dir(model_name),
            input_root,
            float(segment_len),
            int(sample_rate),
            "True" if normalize else "False",
        ),
    )


def _extract_features(model_name, sample_rate, f0_method, include_mutes):
    model_name = _require_name(model_name)
    sliced = os.path.join(_exp_dir(model_name), "data", "sliced_audios")
    if not os.path.isdir(sliced) or not os.listdir(sliced):
        raise gr.Error("Нет нарезанных сегментов — сначала выполните шаг 1.")
    yield from run_job(
        f"Извлечение признаков «{model_name}»",
        command(
            "rvc.training.preprocess.preparing_data",
            _exp_dir(model_name),
            f0_method,
            int(sample_rate),
            int(include_mutes),
        ),
    )


def _train_index(model_name):
    model_name = _require_name(model_name)
    features = os.path.join(_exp_dir(model_name), "data", "features")
    if not os.path.isdir(features) or not os.listdir(features):
        raise gr.Error("Нет признаков — сначала выполните шаг 2.")
    yield from run_job(
        f"Индекс «{model_name}»",
        command("rvc.training.preprocess.extract_index", _exp_dir(model_name), "Auto"),
    )


def _train_model(
    model_name,
    sample_rate,
    total_epoch,
    save_every_epoch,
    batch_size,
    vocoder,
    optimizer,
    gpus,
    pretrain_g,
    pretrain_d,
    save_to_zip,
    save_half,
):
    model_name = _require_name(model_name)
    filelist = os.path.join(_exp_dir(model_name), "data", "filelist.txt")
    if not os.path.isfile(filelist):
        raise gr.Error("Нет filelist.txt — сначала выполните шаги 1 и 2.")
    cmd = command(
        "rvc.training.train",
        "--experiment_dir",
        LOGS_DIR,
        "--model_name",
        model_name,
        "--total_epoch",
        int(total_epoch),
        "--save_every_epoch",
        int(save_every_epoch),
        "--batch_size",
        int(batch_size),
        "--sample_rate",
        int(sample_rate),
        "--vocoder",
        vocoder,
        "--optimizer",
        optimizer,
        "--gpus",
        (gpus or "0").strip(),
        "--save_to_zip",
        "true" if save_to_zip else "false",
        "--save_half",
        "true" if save_half else "false",
    )
    for flag, path in (("--pretrain_g", pretrain_g), ("--pretrain_d", pretrain_d)):
        if path and str(path).strip():
            if not os.path.isfile(str(path).strip()):
                raise gr.Error(f"Претрейн не найден: {path}")
            cmd += [flag, str(path).strip()]
    yield from run_job(f"Обучение «{model_name}»", cmd)


def training_tab():
    with gr.Row():
        model_name = gr.Textbox(label="Имя модели", placeholder="MyVoice", scale=3)
        sample_rate = gr.Dropdown(SAMPLE_RATES, value=48000, label="Частота (Hz)", scale=1)

    with gr.Group():
        gr.Markdown("**Шаг 1 · Нарезка датасета**")
        with gr.Row():
            dataset_files = gr.File(label="Аудиофайлы", file_count="multiple")
            with gr.Column():
                dataset_folder = gr.Textbox(label="…или папка с датасетом", placeholder="/путь/к/аудио")
                with gr.Row():
                    segment_len = gr.Slider(minimum=1.0, maximum=10.0, step=0.1, value=3.0, label="Сегмент (сек)")
                    normalize = gr.Checkbox(True, label="Нормализация")
        slice_btn = gr.Button("Нарезать", variant="primary")

    with gr.Group():
        gr.Markdown("**Шаг 2 · Признаки (F0 + HuBERT)**")
        with gr.Row():
            f0_method = gr.Dropdown(F0_METHODS, value="rmvpe", label="Метод F0")
            include_mutes = gr.Slider(minimum=0, maximum=10, step=1, value=2, label="Мьют-файлов")
            extract_btn = gr.Button("Извлечь", variant="primary")

    with gr.Group():
        gr.Markdown("**Шаг 3 · Обучение**")
        gr.Markdown(
            "Если в папке модели есть `checkpoint.pth`, обучение продолжится с него. "
            "Метрики: `tensorboard --logdir logs`.",
        )
        with gr.Row():
            total_epoch = gr.Slider(minimum=1, maximum=10000, step=1, value=300, label="Всего эпох")
            save_every_epoch = gr.Slider(minimum=1, maximum=100, step=1, value=25, label="Сохранять каждые N")
            batch_size = gr.Slider(minimum=1, maximum=128, step=1, value=8, label="Батч")
        with gr.Accordion("Дополнительно", open=False):
            with gr.Row():
                vocoder = gr.Dropdown(VOCODERS, value="HiFi-GAN", label="Вокодер")
                optimizer = gr.Dropdown(OPTIMIZERS, value="AdamW", label="Оптимизатор")
                gpus = gr.Textbox("0", label="GPU")
            with gr.Row():
                pretrain_g = gr.Textbox(label="Претрейн G (.pth)", placeholder="путь — только для старта с нуля")
                pretrain_d = gr.Textbox(label="Претрейн D (.pth)", placeholder="путь — только для старта с нуля")
            with gr.Row():
                save_to_zip = gr.Checkbox(False, label="Собрать ZIP в конце")
                save_half = gr.Checkbox(True, label="Веса float16")
        with gr.Row():
            train_btn = gr.Button("Обучить", variant="primary")
            index_btn = gr.Button("Построить индекс")
            stop_btn = gr.Button("Остановить", variant="stop")

    log = gr.Textbox(label="Журнал", lines=12, max_lines=12)

    slice_btn.click(
        _slice_dataset,
        inputs=[model_name, dataset_files, dataset_folder, sample_rate, segment_len, normalize],
        outputs=log,
    )
    extract_btn.click(
        _extract_features,
        inputs=[model_name, sample_rate, f0_method, include_mutes],
        outputs=log,
    )
    train_btn.click(
        _train_model,
        inputs=[
            model_name,
            sample_rate,
            total_epoch,
            save_every_epoch,
            batch_size,
            vocoder,
            optimizer,
            gpus,
            pretrain_g,
            pretrain_d,
            save_to_zip,
            save_half,
        ],
        outputs=log,
    )
    index_btn.click(_train_index, inputs=model_name, outputs=log)
    stop_btn.click(request_stop, outputs=log, queue=False, api_name=False)
