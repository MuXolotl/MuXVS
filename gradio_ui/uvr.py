"""Вкладка «UVR»: разделение аудио на стемы.

Логика взята из PolUVR (`PolUVR/utils/app.py`) и встроена в проект:
тот же `Separator`, те же модели и параметры, но вместо пяти вкладок —
один компактный вид с переключателем архитектуры.
Требуется установленный PolUVR (даёт `Separator` и каталоги моделей `UVR_resources`).
"""

import gc
import logging
import os
import shutil

import gradio as gr
import torch
from PolUVR.separator import Separator
from UVR_resources import DEMUCS_v4_MODELS, FORMATS, MDX23C_MODELS, MDXNET_MODELS, ROFORMER_MODELS, VR_ARCH_MODELS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AUTOCAST = DEVICE == "cuda"

ARCHES = {
    "Roformer": {"models": ROFORMER_MODELS, "default": "MelBand Roformer Kim | Big Beta v5e FT by Unwa"},
    "MDX23C": {"models": MDX23C_MODELS, "default": "MDX23C InstVoc HQ"},
    "MDX-NET": {"models": MDXNET_MODELS, "default": "UVR-MDX-NET Inst HQ 5"},
    "VR Arch": {"models": VR_ARCH_MODELS, "default": "1_HP-UVR"},
    "Demucs": {"models": DEMUCS_v4_MODELS, "default": "htdemucs_ft"},
}


def _reset_stems():
    return [gr.update(value=None, visible=False) for _ in range(6)]


def _stem_results(results, output_dir):
    outputs = []
    for num in range(6):
        if num < len(results):
            path = os.path.join(output_dir, results[num])
            outputs.append(gr.update(value=path, visible=True, label=f"Стем {num + 1} ({os.path.basename(path)})"))
        else:
            outputs.append(gr.update(visible=False))
    return outputs


def _prepare_output_dir(input_file, output_directory):
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_dir = os.path.join(output_directory, base_name)
    try:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
    except OSError as error:
        raise gr.Error(f"Не удалось создать папку {output_dir}: {error}") from error
    return output_dir


def _switch_arch(arch):
    """Обновляет список моделей и видимую группу параметров."""
    info = ARCHES[arch]
    choices = sorted(info["models"])
    default = info["default"] if info["default"] in info["models"] else (choices[0] if choices else None)
    updates = [gr.update(choices=choices, value=default)]
    updates.extend(gr.update(visible=name == arch) for name in ARCHES)
    return updates


def _separate(
    arch,
    audio_path,
    model_key,
    output_format,
    batch_size,
    norm_threshold,
    amp_threshold,
    # Roformer / MDX23C
    rof_seg_size,
    rof_override_seg_size,
    rof_overlap,
    rof_pitch_shift,
    mdxc_seg_size,
    mdxc_override_seg_size,
    mdxc_overlap,
    mdxc_pitch_shift,
    # MDX-NET
    mdx_hop_length,
    mdx_seg_size,
    mdx_overlap,
    mdx_denoise,
    # VR Arch
    vr_window_size,
    vr_aggression,
    vr_tta,
    vr_post_process,
    vr_post_process_threshold,
    vr_high_end_process,
    # Demucs
    demucs_seg_size,
    demucs_shifts,
    demucs_overlap,
    demucs_segments_enabled,
    # Папки и имена
    model_dir,
    output_dir,
    rename_template,
    progress=gr.Progress(track_tqdm=True),
):
    if not audio_path or not os.path.isfile(audio_path):
        raise gr.Error("Прикрепите аудиофайл.")
    if not model_key:
        raise gr.Error("Выберите модель.")

    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    print(f"\n🎵 UVR [{arch}] — {base_name} — {model_key}")

    if arch == "Roformer":
        model_filename, extra = ROFORMER_MODELS[model_key], {
            "mdxc_params": {
                "segment_size": rof_seg_size,
                "override_model_segment_size": rof_override_seg_size,
                "batch_size": batch_size,
                "overlap": rof_overlap,
                "pitch_shift": rof_pitch_shift,
            },
        }
    elif arch == "MDX23C":
        model_filename, extra = MDX23C_MODELS[model_key], {
            "mdxc_params": {
                "segment_size": mdxc_seg_size,
                "override_model_segment_size": mdxc_override_seg_size,
                "batch_size": batch_size,
                "overlap": mdxc_overlap,
                "pitch_shift": mdxc_pitch_shift,
            },
        }
    elif arch == "MDX-NET":
        model_filename, extra = MDXNET_MODELS[model_key], {
            "mdx_params": {
                "hop_length": mdx_hop_length,
                "segment_size": mdx_seg_size,
                "overlap": mdx_overlap,
                "batch_size": batch_size,
                "enable_denoise": mdx_denoise,
            },
        }
    elif arch == "VR Arch":
        model_filename, extra = VR_ARCH_MODELS[model_key], {
            "vr_params": {
                "batch_size": batch_size,
                "window_size": vr_window_size,
                "aggression": vr_aggression,
                "enable_tta": vr_tta,
                "enable_post_process": vr_post_process,
                "post_process_threshold": vr_post_process_threshold,
                "high_end_process": vr_high_end_process,
            },
        }
    else:
        model_filename, extra = DEMUCS_v4_MODELS[model_key], {
            "demucs_params": {
                "segment_size": demucs_seg_size,
                "shifts": demucs_shifts,
                "overlap": demucs_overlap,
                "segments_enabled": demucs_segments_enabled,
            },
        }

    try:
        yield _reset_stems()
        out_dir = _prepare_output_dir(audio_path, output_dir)
        stem_names = {
            "All Stems": rename_template.replace("NAME", base_name).replace("STEM", "All Stems").replace("MODEL", model_key),
        }

        separator = Separator(
            log_level=logging.WARNING,
            model_file_dir=model_dir,
            output_dir=out_dir,
            output_format=output_format,
            normalization_threshold=norm_threshold,
            amplification_threshold=amp_threshold,
            use_autocast=USE_AUTOCAST,
            **extra,
        )
        progress(0.2, desc="Загрузка модели…")
        separator.load_model(model_filename=model_filename)
        progress(0.7, desc="Разделение аудио…")
        results = separator.separate(audio_path, stem_names)
        print(f"Готово: {', '.join(results)}")
        yield _stem_results(results, out_dir)
    except Exception as error:
        raise gr.Error(f"Ошибка разделения ({arch}): {error}") from error
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


def uvr_tab(models_dir: str, output_dir: str):
    with gr.Row():
        arch = gr.Dropdown(sorted(ARCHES), value="Roformer", label="Архитектура", scale=1)
        roformer_default = ARCHES["Roformer"]["default"]
        model = gr.Dropdown(
            sorted(ROFORMER_MODELS),
            value=roformer_default if roformer_default in ROFORMER_MODELS else None,
            label="Модель",
            scale=3,
        )
        output_format = gr.Dropdown(FORMATS, value="wav", label="Формат", scale=1)

    input_audio = gr.Audio(label="Исходное аудио", type="filepath")

    with gr.Accordion("Параметры", open=False):
        with gr.Row():
            batch_size = gr.Slider(minimum=1, maximum=16, step=1, value=1, label="Батч")
            norm_threshold = gr.Slider(minimum=0.1, maximum=1, step=0.1, value=0.9, label="Нормализация")
            amp_threshold = gr.Slider(minimum=0.0, maximum=1, step=0.1, value=0.0, label="Усиление тихих мест")

        with gr.Group(visible=True) as roformer_group:
            rof_override = gr.Checkbox(False, label="Свой размер сегмента")
            with gr.Row():
                rof_seg = gr.Slider(minimum=32, maximum=4000, step=32, value=256, label="Сегмент", visible=False)
                rof_overlap = gr.Slider(minimum=2, maximum=10, step=1, value=8, label="Перекрытие")
                rof_pitch = gr.Slider(minimum=-24, maximum=24, step=1, value=0, label="Сдвиг тона")

        with gr.Group(visible=False) as mdxc_group:
            mdxc_override = gr.Checkbox(False, label="Свой размер сегмента")
            with gr.Row():
                mdxc_seg = gr.Slider(minimum=32, maximum=4000, step=32, value=256, label="Сегмент", visible=False)
                mdxc_overlap = gr.Slider(minimum=2, maximum=50, step=1, value=8, label="Перекрытие")
                mdxc_pitch = gr.Slider(minimum=-24, maximum=24, step=1, value=0, label="Сдвиг тона")

        with gr.Group(visible=False) as mdx_group:
            mdx_denoise = gr.Checkbox(False, label="Шумоподавление после разделения")
            with gr.Row():
                mdx_hop = gr.Slider(minimum=32, maximum=2048, step=32, value=1024, label="Hop Length")
                mdx_seg = gr.Slider(minimum=32, maximum=4000, step=32, value=256, label="Сегмент")
                mdx_overlap = gr.Slider(minimum=0.001, maximum=0.999, step=0.001, value=0.25, label="Перекрытие")

        with gr.Group(visible=False) as vr_group:
            with gr.Row():
                vr_post = gr.Checkbox(False, label="Пост-обработка")
                vr_tta = gr.Checkbox(False, label="TTA")
                vr_high_end = gr.Checkbox(False, label="Восстановить верха")
            with gr.Row():
                vr_post_thr = gr.Slider(minimum=0.1, maximum=0.3, step=0.1, value=0.2, label="Порог пост-обработки", visible=False)
                vr_window = gr.Slider(minimum=320, maximum=1024, step=32, value=512, label="Окно")
                vr_aggr = gr.Slider(minimum=1, maximum=100, step=1, value=5, label="Агрессия")

        with gr.Group(visible=False) as demucs_group:
            demucs_segen = gr.Checkbox(True, label="Обработка по сегментам")
            with gr.Row():
                demucs_seg = gr.Slider(minimum=1, maximum=100, step=1, value=40, label="Сегмент")
                demucs_overlap = gr.Slider(minimum=0.001, maximum=0.999, step=0.001, value=0.25, label="Перекрытие")
                demucs_shifts = gr.Slider(minimum=0, maximum=20, step=1, value=2, label="Сдвиги")

    with gr.Accordion("Папки и имена файлов", open=False):
        with gr.Row():
            model_dir = gr.Textbox(models_dir, label="Папка моделей")
            output_dir_box = gr.Textbox(output_dir, label="Папка результата")
        rename_template = gr.Textbox(
            "NAME_(STEM)_MODEL",
            label="Шаблон имён",
            info="Ключи: NAME — файл, STEM — стем, MODEL — модель.",
        )

    separate_btn = gr.Button("Разделить", variant="primary")

    stems = []
    with gr.Column():
        for pair in range(0, 6, 2):
            with gr.Row():
                stems.append(gr.Audio(visible=False, interactive=False, label=f"Стем {pair + 1}"))
                stems.append(gr.Audio(visible=False, interactive=False, label=f"Стем {pair + 2}"))

    groups = [roformer_group, mdxc_group, mdx_group, vr_group, demucs_group]
    arch.change(_switch_arch, inputs=arch, outputs=[model, *groups], api_name=False)
    rof_override.change(lambda v: gr.update(visible=v), inputs=rof_override, outputs=rof_seg, api_name=False)
    mdxc_override.change(lambda v: gr.update(visible=v), inputs=mdxc_override, outputs=mdxc_seg, api_name=False)
    vr_post.change(lambda v: gr.update(visible=v), inputs=vr_post, outputs=vr_post_thr, api_name=False)

    separate_btn.click(
        _separate,
        inputs=[
            arch,
            input_audio,
            model,
            output_format,
            batch_size,
            norm_threshold,
            amp_threshold,
            rof_seg,
            rof_override,
            rof_overlap,
            rof_pitch,
            mdxc_seg,
            mdxc_override,
            mdxc_overlap,
            mdxc_pitch,
            mdx_hop,
            mdx_seg,
            mdx_overlap,
            mdx_denoise,
            vr_window,
            vr_aggr,
            vr_tta,
            vr_post,
            vr_post_thr,
            vr_high_end,
            demucs_seg,
            demucs_shifts,
            demucs_overlap,
            demucs_segen,
            model_dir,
            output_dir_box,
            rename_template,
        ],
        outputs=stems,
    )
