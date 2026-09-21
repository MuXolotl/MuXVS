"""Вкладка «Конвертация»: замена голоса в одном файле или пакетно."""

import gradio as gr

from gradio_ui.common import F0_METHODS, OUTPUT_FORMATS, RVC_OUTPUT_DIR, conversion_settings, model_row


def _convert(
    rvc_model,
    input_path,
    f0_method,
    rvc_pitch,
    output_format,
    index_rate,
    protect,
    volume_envelope,
    f0_min,
    f0_max,
    autopitch,
    autopitch_threshold,
    autotune,
    autotune_tonic,
    autotune_scale,
    autotune_strength,
    stereo_sound,
    audio_upscaling,
    progress=gr.Progress(track_tqdm=True),
):
    from rvc.inference.infer import rvc_infer

    return rvc_infer(
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
        progress=progress,
    )


def _convert_batch(
    rvc_model,
    dir_input,
    files,
    output_dir,
    f0_method,
    rvc_pitch,
    output_format,
    index_rate,
    protect,
    volume_envelope,
    f0_min,
    f0_max,
    autopitch,
    autopitch_threshold,
    autotune,
    autotune_tonic,
    autotune_scale,
    autotune_strength,
    stereo_sound,
    audio_upscaling,
    progress=gr.Progress(track_tqdm=True),
):
    from rvc.inference.infer import rvc_batch_infer

    return rvc_batch_infer(
        rvc_model=rvc_model,
        dir_input=dir_input,
        files=files,
        output_dir=output_dir,
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
        progress=progress,
    )


def conversion_tab():
    rvc_model = model_row()

    with gr.Row():
        with gr.Column(scale=1):
            input_audio = gr.Audio(label="Исходное аудио", type="filepath")
            with gr.Row():
                rvc_pitch = gr.Slider(minimum=-24, maximum=24, step=1, value=0, label="Тон (полутоны)")
                f0_method = gr.Dropdown(F0_METHODS, value="rmvpe", label="Метод F0")
        with gr.Column(scale=1):
            convert_btn = gr.Button("Конвертировать", variant="primary")
            output_audio = gr.Audio(label="Результат", interactive=False)
            output_format = gr.Dropdown(OUTPUT_FORMATS, value="mp3", label="Формат", scale=1)

    settings = conversion_settings(pitch=rvc_pitch)

    with gr.Accordion("Пакетная конвертация", open=False):
        gr.Markdown("Папка и/или несколько файлов. Используются модель и настройки выше.")
        with gr.Row():
            dir_input = gr.Textbox(label="Папка с аудио", placeholder="/путь/к/папке", scale=2)
            output_dir = gr.Textbox(label="Папка результата", value=RVC_OUTPUT_DIR, scale=2)
        batch_files = gr.File(label="…или прикрепите файлы", file_count="multiple", height=180)
        with gr.Row(equal_height=True):
            batch_btn = gr.Button("Конвертировать пакет", variant="primary", scale=1)
            batch_info = gr.Textbox(label="Результат", lines=6, scale=2)

    shared_inputs = [
        f0_method,
        rvc_pitch,
        output_format,
        settings["index_rate"],
        settings["protect"],
        settings["volume_envelope"],
        settings["f0_min"],
        settings["f0_max"],
        settings["autopitch"],
        settings["autopitch_threshold"],
        settings["autotune"],
        settings["autotune_tonic"],
        settings["autotune_scale"],
        settings["autotune_strength"],
        settings["stereo_sound"],
        settings["audio_upscaling"],
    ]
    convert_btn.click(_convert, inputs=[rvc_model, input_audio, *shared_inputs], outputs=output_audio)
    batch_btn.click(
        _convert_batch,
        inputs=[rvc_model, dir_input, batch_files, output_dir, *shared_inputs],
        outputs=batch_info,
    )
