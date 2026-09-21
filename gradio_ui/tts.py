"""Вкладка «TTS»: синтез речи из текста + замена голоса."""

import gradio as gr

from gradio_ui.common import EDGE_VOICES, F0_METHODS, OUTPUT_FORMATS, conversion_settings, model_row, update_edge_voices


def _synthesize(
    rvc_model,
    tts_voice,
    tts_text,
    tts_rate,
    tts_volume,
    tts_pitch,
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
    from rvc.inference.infer import rvc_edgetts_infer

    return rvc_edgetts_infer(
        rvc_model=rvc_model,
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
        tts_voice=tts_voice,
        tts_text=tts_text,
        tts_rate=tts_rate,
        tts_volume=tts_volume,
        tts_pitch=tts_pitch,
        audio_upscaling=audio_upscaling,
        progress=progress,
    )


def tts_tab():
    rvc_model = model_row()

    with gr.Row():
        language = gr.Dropdown(sorted(EDGE_VOICES), label="Язык")
        tts_voice = gr.Dropdown(
            EDGE_VOICES["Английский (Великобритания)"],
            value="en-GB-SoniaNeural",
            label="Голос",
        )
    language.change(update_edge_voices, inputs=language, outputs=tts_voice, api_name=False)

    tts_text = gr.Textbox(label="Текст", lines=4, placeholder="Введите текст для озвучки…")

    with gr.Row():
        rvc_pitch = gr.Slider(minimum=-24, maximum=24, step=1, value=0, label="Тон (полутоны)")
        f0_method = gr.Dropdown(F0_METHODS, value="rmvpe", label="Метод F0")
        output_format = gr.Dropdown(OUTPUT_FORMATS, value="mp3", label="Формат")

    with gr.Accordion("Параметры речи", open=False):
        with gr.Row():
            tts_rate = gr.Slider(minimum=-100, maximum=100, step=1, value=0, label="Скорость")
            tts_volume = gr.Slider(minimum=-100, maximum=100, step=1, value=0, label="Громкость")
            tts_pitch = gr.Slider(minimum=-100, maximum=100, step=1, value=0, label="Тон речи")

    settings = conversion_settings(pitch=rvc_pitch)

    convert_btn = gr.Button("Озвучить", variant="primary")
    with gr.Row():
        synth_audio = gr.Audio(label="Синтез TTS", interactive=False)
        output_audio = gr.Audio(label="Результат", interactive=False)

    convert_btn.click(
        _synthesize,
        inputs=[
            rvc_model,
            tts_voice,
            tts_text,
            tts_rate,
            tts_volume,
            tts_pitch,
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
        ],
        outputs=[synth_audio, output_audio],
    )
