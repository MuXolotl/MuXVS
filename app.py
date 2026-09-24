import logging
import os
import sys
import traceback
import warnings
from typing import Any

import torch

# Configuring the environment and logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Disable unnecessary TensorFlow logs
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"  # Disabling Gradio analytics
logging.basicConfig(level=logging.WARNING)  # Disable all logs, except WARNING and above
warnings.filterwarnings("ignore")  # Disable all warnings

if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

import gradio as gr

from assets.model_installer import check_and_install_models
from assets.notebook_check import colab_check, kaggle_check
from assets.version import __version__, __version_info__
from gradio_ui.common import UVR_MODELS_DIR, UVR_OUTPUT_DIR, model_dropdowns, refresh_models
from gradio_ui.inference import conversion_tab
from gradio_ui.models import models_tab
from gradio_ui.tools import tools_tab
from gradio_ui.training import training_tab

# Constants
DEFAULT_SERVER_NAME = "127.0.0.1"
DEFAULT_PORT = 4000
MAX_PORT_ATTEMPTS = 10

APP_CSS = """
footer{display:none !important}
.gradio-container{width:100% !important;max-width:1400px !important;margin:0 auto !important}
.app-header{text-align:center !important}
.tab-container{justify-content:center !important}
@media (max-width:768px){
.tab-container{justify-content:flex-start !important;overflow-x:auto !important;max-width:100% !important}
.tab-container>button{flex-shrink:0 !important}
}
"""

RUN_FROM_JUPYTER_NOTEBOOKS = colab_check() or kaggle_check()


def check_uvr() -> tuple[str, str, str, Any]:
    """Проверяет, доступен ли встроенный UVR (нужен установленный PolUVR)."""
    try:
        from gradio_ui.uvr import uvr_tab

        return "UVR", "", "", uvr_tab
    except Exception:
        return (
            "UVR ⚠️",
            "UVR недоступен: не установлен PolUVR или его зависимости.",
            traceback.format_exc(),
            None,
        )


uvr_title, uvr_message, uvr_error, uvr_ui = check_uvr()


def is_offline_mode() -> bool:
    return "--offline" in sys.argv


def get_title() -> str:
    """Формирует заголовок окна с версией."""
    base_title = f"MuXVS v{__version__} - Politrees"
    if is_offline_mode():
        return f"{base_title} (offline)"
    return base_title


# Gradio Interface
with gr.Blocks(
    title=get_title(),
    css=APP_CSS,
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="slate",
        neutral_hue="neutral",
        spacing_size="sm",
        radius_size="lg",
    ),
) as MuXVS:
    gr.Markdown(
        elem_classes=["app-header"],
        value=f"# MuXVS <small>v{__version__}</small>\n"
        "[Telegram](https://t.me/politrees) · "
        "[Чат](https://t.me/+GMTP7hZqY0E4OGRi) · "
        "[YouTube](https://www.youtube.com/@Politrees) · "
        "[GitHub](https://github.com/MuXolotl/MuXVS)",
    )

    with gr.Tab("Конвертация") as conversion_tab_ui:
        conversion_tab(include_tts=not is_offline_mode())
    # список обновляем при каждом открытии вкладки, а не только кнопкой.
    conversion_tab_ui.select(refresh_models, outputs=model_dropdowns(), api_name=False)

    # Обучение только при наличии CUDA
    if torch.cuda.is_available():
        with gr.Tab("Обучение"):
            training_tab()

    with gr.Tab(uvr_title):
        if uvr_ui is not None:
            uvr_ui(UVR_MODELS_DIR, UVR_OUTPUT_DIR)
        else:
            gr.HTML(f"<center><h2>{uvr_message}</h2></center>")
            gr.Code(value=uvr_error, language="python", interactive=False, show_label=False)

    with gr.Tab("Модели"):
        models_tab(is_offline_mode())

    with gr.Tab("Инструменты"):
        tools_tab()


def launch_gradio(server_name: str, server_port: int) -> None:
    MuXVS.launch(
        favicon_path="assets/logo.ico",
        inbrowser=not RUN_FROM_JUPYTER_NOTEBOOKS,
        share=RUN_FROM_JUPYTER_NOTEBOOKS and ("--no-share" not in sys.argv),
        quiet=RUN_FROM_JUPYTER_NOTEBOOKS,
        server_name=server_name,
        server_port=server_port,
        show_error=True,
        debug=True,
    )


def get_value_from_args(key: str, default: Any = None) -> Any:
    if key in sys.argv:
        index = sys.argv.index(key) + 1
        if index < len(sys.argv):
            return sys.argv[index]
    return default


if __name__ == "__main__":
    print("Среда запуска: ", "Jupyter Notebook" if RUN_FROM_JUPYTER_NOTEBOOKS else "LocalHost")

    # Красивый вывод версии
    print(f"\n╔{'═' * 42}╗")
    print(f"║{'MuXVS v' + __version__:^42}║")
    if __version_info__["is_prerelease"]:
        print(f"║{'[!] Pre-release версия':^42}║")
    print(f"╚{'═' * 42}╝\n")

    if uvr_ui is None:
        print("⚠️ [UVR] Импорт не удался, вкладка UVR будет отключена!")

    print("Запуск интерфейса MuXVS. Подождите...")
    if is_offline_mode():
        # В оффлайне сети заведомо нет: проверка только сыпала бы ошибками загрузки.
        print("Оффлайн-режим: проверка и загрузка моделей пропущены.")
    else:
        check_and_install_models()  # Checking and installing models

    port = int(get_value_from_args("--port", DEFAULT_PORT))
    server = get_value_from_args("--server-name", DEFAULT_SERVER_NAME)

    for _ in range(MAX_PORT_ATTEMPTS):
        try:
            launch_gradio(server, port)
            break
        except OSError:
            print(f"Не удалось запустить на порту {port}, повторите попытку на порту {port - 1}...")
            port -= 1
        except Exception as error:
            print(f"Произошла ошибка при запуске Gradio: {error}")
            break
