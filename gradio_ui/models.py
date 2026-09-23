"""Вкладка «Модели»: загрузка RVC-моделей и установка эмбеддеров."""

import os
import shutil
import urllib.parse
import urllib.request

import gradio as gr

from gradio_ui.common import PROJECT_ROOT, model_dropdowns, refresh_models

EMBEDDERS_DIR = os.path.join(PROJECT_ROOT, "assets", "models", "embedders")
EMBEDDER_PATH = os.path.join(EMBEDDERS_DIR, "contentvec_base.pt")
EMBEDDER_BASE_URL = "https://huggingface.co/Politrees/RVC_resources/resolve/main/embedders/pytorch/"
EMBEDDER_MODELS = [
    "contentvec_base.pt",
    "hubert_base.pt",
    "korean_hubert_base.pt",
    "chinese_hubert_base.pt",
    "portuguese_hubert_base.pt",
    "japanese_hubert_base.pt",
]


def _download_from_url(url, dir_name, progress=gr.Progress()):
    from rvc.inference.modules.model_manager import download_from_url

    return download_from_url(url, dir_name, progress=progress)


def _upload_zip(zip_file, dir_name, progress=gr.Progress()):
    from rvc.inference.modules.model_manager import upload_zip_file

    return upload_zip_file(zip_file, dir_name, progress=progress)


def _upload_files(pth_file, index_file, dir_name, progress=gr.Progress()):
    from rvc.inference.modules.model_manager import upload_separate_files

    return upload_separate_files(pth_file, index_file, dir_name, progress=progress)


def _install_embedder(model_name, custom_url, progress=gr.Progress()):
    try:
        if custom_url:
            if not custom_url.endswith((".pt", "?download=true")):
                return "Ошибка: URL должен вести к .pt файлу или заканчиваться на '?download=true'."
            if urllib.parse.urlparse(custom_url).netloc != "huggingface.co":
                return "Ошибка: разрешён только huggingface.co."
            model_url, model_name = custom_url, os.path.basename(urllib.parse.urlparse(custom_url).path)
        else:
            model_url = EMBEDDER_BASE_URL + model_name

        tmp_path = os.path.join(EMBEDDERS_DIR, "tmp_model.pt")
        progress(0.4, desc=f'[~] Установка "{model_name}"…')
        with urllib.request.urlopen(model_url) as response, open(tmp_path, "wb") as out:
            shutil.copyfileobj(response, out)

        progress(0.8, desc="[~] Замена эмбеддера…")
        if os.path.exists(EMBEDDER_PATH):
            os.remove(EMBEDDER_PATH)
        os.rename(tmp_path, EMBEDDER_PATH)
        return f'Модель "{model_name}" установлена.'
    except Exception as error:
        return f'Ошибка установки "{model_name}": {error}'


def _toggle_custom_embedder(enabled: bool):
    if enabled:
        return gr.update(visible=True, value=""), gr.update(visible=False)
    return gr.update(visible=False, value=""), gr.update(visible=True, value="contentvec_base.pt")


def models_tab(offline: bool):
    message = gr.Textbox(label="Результат", interactive=False)

    # Единственная кнопка обновления списка: остальные вкладки подхватывают
    # изменения сами — после загрузки модели и при открытии вкладки.
    with gr.Group(), gr.Row(equal_height=True):
        refresh_btn = gr.Button("Обновить список моделей", variant="secondary", scale=1)
        gr.Markdown("Список голосовых моделей обновляется сам: после загрузки и при открытии вкладки «Конвертация».")
    refresh_btn.click(refresh_models, outputs=model_dropdowns(), api_name=False)

    if not offline:
        with gr.Accordion("По ссылке (ZIP)", open=True):
            gr.Markdown("HuggingFace · Pixeldrain · Google Drive · Mega · Яндекс Диск · Dropbox")
            with gr.Row():
                zip_link = gr.Textbox(label="Ссылка", scale=3)
                url_name = gr.Textbox(label="Имя модели", scale=2)
            url_btn = gr.Button("Скачать", variant="primary")
            url_btn.click(_download_from_url, inputs=[zip_link, url_name], outputs=message).then(
                refresh_models,
                outputs=model_dropdowns(),
                api_name=False,
            )

    with gr.Accordion("ZIP-файл", open=False):
        with gr.Row():
            zip_file = gr.File(label="ZIP", file_types=[".zip"], file_count="single", scale=3)
            zip_name = gr.Textbox(label="Имя модели", scale=2)
        zip_btn = gr.Button("Загрузить", variant="primary")
        zip_btn.click(_upload_zip, inputs=[zip_file, zip_name], outputs=message).then(
            refresh_models,
            outputs=model_dropdowns(),
            api_name=False,
        )

    with gr.Accordion("Файлы .pth + .index", open=False):
        with gr.Row():
            pth_file = gr.File(label=".pth", file_types=[".pth"], file_count="single")
            index_file = gr.File(label=".index", file_types=[".index"], file_count="single")
            files_name = gr.Textbox(label="Имя модели")
        files_btn = gr.Button("Загрузить", variant="primary")
        files_btn.click(_upload_files, inputs=[pth_file, index_file, files_name], outputs=message).then(
            refresh_models,
            outputs=model_dropdowns(),
            api_name=False,
        )

    if not offline:
        with gr.Accordion("Эмбеддер", open=False):
            gr.Markdown("Менять нужно, только если обучали модель с другим эмбеддером.")
            with gr.Row():
                custom_check = gr.Checkbox(False, label="Свой URL")
                custom_url = gr.Textbox(label="URL модели", visible=False, scale=3)
                embedder_drop = gr.Dropdown(EMBEDDER_MODELS, value="contentvec_base.pt", label="Эмбеддер", scale=3)
            embedder_btn = gr.Button("Установить", variant="primary")
            custom_check.change(
                _toggle_custom_embedder,
                inputs=custom_check,
                outputs=[custom_url, embedder_drop],
                api_name=False,
            )
            embedder_btn.click(_install_embedder, inputs=[embedder_drop, custom_url], outputs=message)
