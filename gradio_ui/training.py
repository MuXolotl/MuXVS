"""Вкладка «Обучение»: секции как в официальном RVC WebUI.

Нарезка датасета → Признаки + индекс → Обучение.
Секции запускаются отдельными процессами с общим живым журналом.
"""

import os
import shutil

import gradio as gr

from assets.model_installer import NO_PRETRAIN, PRETRAIN_CHOICES, ensure_pretrains
from gradio_ui.jobs import command, request_stop, run_job

LOGS_DIR = os.path.join(os.getcwd(), "logs")
SAMPLE_RATES = [32000, 40000, 48000]
VOCODERS = ["HiFi-GAN", "MRF HiFi-GAN", "RefineGAN"]
OPTIMIZERS = ["AdamW", "AdaBelief", "PolOpt"]
F0_METHODS = ["rmvpe", "rmvpe+", "hpa-rmvpe"]
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac", ".wma", ".aiff", ".webm", ".mp4")


def _section_title(text: str):
    """Заголовок секции с нормальными отступами (Markdown липнет к краям группы)."""
    return gr.HTML(f"<div style='font-size:14px;font-weight:600;text-align:center;padding:2px 4px 6px;'>{text}</div>")


def _section_hint(code1: str, code2: str):
    """Центрированная подсказка под заголовком секции обучения."""
    return gr.HTML(
        "<div style='font-size:13px;opacity:0.85;text-align:center;padding:0 4px 8px;'>"
        f"Если в папке модели есть <code>{code1}</code>, обучение продолжится с него. "
        f"Метрики: <code>{code2}</code>.</div>",
    )


def _exp_dir(model_name: str) -> str:
    return os.path.join(LOGS_DIR, model_name.strip())


def _require_name(model_name: str) -> str:
    if not model_name or not model_name.strip():
        raise gr.Error("Укажите имя модели.")
    return model_name.strip()


def _input_root(model_name: str, files, folder: str) -> str:
    """Папка входа для нарезки: указанная папка или загруженные файлы."""
    if folder and folder.strip():
        if not os.path.isdir(folder.strip()):
            raise gr.Error(f"Папка не найдена: {folder.strip()}")
        return folder.strip()
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
    raise gr.Error("Укажите папку с датасетом или прикрепите аудиофайлы.")


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
        raise gr.Error("Нет нарезанных сегментов — сначала выполните нарезку.")
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
        raise gr.Error("Нет признаков — сначала извлеките их.")
    yield from run_job(
        f"Индекс «{model_name}»",
        command("rvc.training.preprocess.extract_index", _exp_dir(model_name), "Auto"),
    )


def _resolve_pretrains(pretrain, pretrain_g, pretrain_d, sample_rate, exp_dir):
    """Пути претрейнов G/D: свои файлы, встроенный набор или ничего.

    Генератор: отдаёт строки журнала, возвращает (путь G, путь D).
    Свои файлы имеют приоритет над встроенным набором. При продолжении
    обучения с чекпоинта претрейны не нужны — скачивание пропускается.
    """
    manual_g = (pretrain_g or "").strip()
    manual_d = (pretrain_d or "").strip()
    if manual_g or manual_d:
        if not (manual_g and manual_d):
            raise gr.Error("Укажите оба своих претрейна (G и D) или ни одного.")
        for path in (manual_g, manual_d):
            if not os.path.isfile(path):
                raise gr.Error(f"Претрейн не найден: {path}")
        yield f"• Свои претрейны: {os.path.basename(manual_g)}, {os.path.basename(manual_d)}"
        return manual_g, manual_d
    if os.path.isfile(os.path.join(exp_dir, "checkpoint.pth")):
        yield "• Найден checkpoint.pth — обучение продолжится, претрейны не нужны."
        return None, None
    if pretrain == NO_PRETRAIN:
        yield "• Без претрейна: потребуется больше эпох и данных."
        return None, None
    try:
        return (yield from ensure_pretrains(pretrain, sample_rate))
    except ValueError as error:
        raise gr.Error(str(error)) from error


def _drain(generator):
    """Прогоняет генератор до конца, показывая накопленный журнал.

    Возвращает (строки, return-значение генератора).
    """
    lines = []
    try:
        while True:
            lines.append(next(generator))
            yield "\n".join(lines)
    except StopIteration as done:
        return lines, done.value


def _train_model(
    model_name,
    sample_rate,
    total_epoch,
    save_every_epoch,
    batch_size,
    vocoder,
    optimizer,
    gpus,
    pretrain,
    pretrain_g,
    pretrain_d,
    save_to_zip,
    save_half,
):
    model_name = _require_name(model_name)
    exp_dir = _exp_dir(model_name)
    if not os.path.isfile(os.path.join(exp_dir, "data", "filelist.txt")):
        raise gr.Error("Нет filelist.txt — сначала выполните нарезку и извлечение признаков.")
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

    notes, (resolved_g, resolved_d) = yield from _drain(
        _resolve_pretrains(pretrain, pretrain_g, pretrain_d, int(sample_rate), exp_dir),
    )
    if resolved_g and resolved_d:
        cmd += ["--pretrain_g", resolved_g, "--pretrain_d", resolved_d]
    yield from run_job(f"Обучение «{model_name}»", cmd, prefix="\n".join(notes))


def training_tab():
    model_name = gr.Textbox(label="Имя модели", placeholder="MyVoice")

    with gr.Accordion("Подготовка данных", open=False):
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                dataset_folder = gr.Textbox(label="Папка с датасетом", placeholder="/путь/к/аудио")
                dataset_files = gr.File(label="…или загрузите файлы", file_count="multiple", height=260)
                slice_btn = gr.Button("Нарезать", variant="primary")
            with gr.Column(scale=1):
                _section_title("Настройки обработки")
                segment_len = gr.Slider(minimum=1.0, maximum=10.0, step=0.1, value=3.0, label="Сегмент (сек)")
                with gr.Row(equal_height=True):
                    normalize = gr.Checkbox(value=True, label="Нормализация")
                    sample_rate = gr.Dropdown(SAMPLE_RATES, value=48000, label="Частота (Hz)")
                _section_title("Признаки (F0 + HuBERT)")
                with gr.Row(equal_height=True):
                    f0_method = gr.Dropdown(F0_METHODS, value="rmvpe", label="Метод F0")
                    include_mutes = gr.Slider(minimum=0, maximum=10, step=1, value=2, label="Мьют-файлов")
                extract_btn = gr.Button("Извлечь", variant="primary")
                index_btn = gr.Button("Построить индекс")

    _section_title("Обучение")
    _section_hint("checkpoint.pth", "tensorboard --logdir logs")
    with gr.Row(equal_height=True):
        total_epoch = gr.Slider(minimum=1, maximum=10000, step=1, value=300, label="Всего эпох")
        save_every_epoch = gr.Slider(minimum=1, maximum=100, step=1, value=25, label="Сохранять каждые N")
        batch_size = gr.Slider(minimum=1, maximum=128, step=1, value=8, label="Батч")
    with gr.Accordion("Дополнительно", open=False):
        with gr.Row(equal_height=True):
            vocoder = gr.Dropdown(VOCODERS, value="HiFi-GAN", label="Вокодер")
            optimizer = gr.Dropdown(OPTIMIZERS, value="AdamW", label="Оптимизатор")
            gpus = gr.Textbox("0", label="GPU")
        pretrain = gr.Dropdown(
            PRETRAIN_CHOICES,
            value="Default",
            label="Претрейн",
            info="Встроенный набор скачивается сам при старте обучения.",
        )
        with gr.Row(equal_height=True):
            pretrain_g = gr.Textbox(label="Свой претрейн G", placeholder="Путь к .pth — вместо встроенного")
            pretrain_d = gr.Textbox(label="Свой претрейн D", placeholder="Путь к .pth — вместо встроенного")
        with gr.Row(equal_height=True):
            save_to_zip = gr.Checkbox(False, label="Собрать ZIP в конце")
            save_half = gr.Checkbox(True, label="Веса float16")
    with gr.Row(equal_height=True):
        train_btn = gr.Button("Запустить обучение", variant="primary")
        stop_btn = gr.Button("Завершить процесс", variant="stop")

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
            pretrain,
            pretrain_g,
            pretrain_d,
            save_to_zip,
            save_half,
        ],
        outputs=log,
    )
    index_btn.click(_train_index, inputs=model_name, outputs=log)
    stop_btn.click(request_stop, outputs=log, queue=False, api_name=False)
