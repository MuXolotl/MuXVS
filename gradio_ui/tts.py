"""Вкладка «TTS»: синтез речи из текста + замена голоса (классический вид)."""

import gradio as gr

from gradio_ui.common import (
    EDGE_VOICES,
    OUTPUT_FORMATS,
    conversion_settings,
    model_select,
    pitch_group,
    update_edge_voices,
)

DEFAULT_LANGUAGE = "Английский (Великобритания)"


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
    with gr.Row():
        with gr.Column(variant="panel", scale=1):
            rvc_model = model_select()
            with gr.Group():
                language = gr.Dropdown(
                    value=DEFAULT_LANGUAGE,
                    label="Язык",
                    choices=list(EDGE_VOICES),
                )
                tts_voice = gr.Dropdown(
                    value="en-GB-SoniaNeural",
                    label="Голос",
                    choices=EDGE_VOICES[DEFAULT_LANGUAGE],
                )
        with gr.Column(variant="panel", scale=2):
            with gr.Column(), gr.Group():
                autopitch, autopitch_threshold, rvc_pitch = pitch_group()
            synth_voice = gr.Audio(
                label="Синтезированный TTS голос",
                show_download_button=True,
                interactive=False,
            )

    with gr.Accordion("Настройки синтеза речи", open=False), gr.Group(), gr.Row():
        tts_pitch = gr.Slider(
            minimum=-100,
            maximum=100,
            step=1,
            value=0,
            label="Регулировка высоты тона TTS",
            info="-100 - мужской голос | 100 - женский голос",
        )
        tts_volume = gr.Slider(
            minimum=-100,
            maximum=100,
            step=1,
            value=0,
            label="Громкость речи",
            info="Громкость воспроизведения синтеза речи",
        )
        tts_rate = gr.Slider(
            minimum=-100,
            maximum=100,
            step=1,
            value=0,
            label="Скорость речи",
            info="Скорость воспроизведения синтеза речи",
        )

    tts_text = gr.Textbox(label="Введите текст", lines=5)

    with gr.Group(), gr.Row(equal_height=True):
        generate_btn = gr.Button(
            value="Генерировать",
            variant="primary",
            scale=2,
        )
        converted_synth_voice = gr.Audio(
            label="Преобразованный TTS голос",
            show_download_button=True,
            interactive=False,
            scale=9,
        )
        with gr.Column(min_width=160):
            output_format = gr.Dropdown(
                value="mp3",
                label="Формат файла",
                choices=OUTPUT_FORMATS,
            )

    settings = conversion_settings()

    # Обновление списка TTS-голосов
    language.change(update_edge_voices, inputs=language, outputs=tts_voice, api_name=False)

    # Обновление списка моделей — внутри model_select(); авто-тон — внутри pitch_group().

    # Запуск процесса преобразования
    generate_btn.click(
        _synthesize,
        inputs=[
            rvc_model,
            tts_voice,
            tts_text,
            tts_rate,
            tts_volume,
            tts_pitch,
            settings["f0_method"],
            rvc_pitch,
            output_format,
            settings["index_rate"],
            settings["protect"],
            settings["volume_envelope"],
            settings["f0_min"],
            settings["f0_max"],
            autopitch,
            autopitch_threshold,
            settings["autotune"],
            settings["autotune_tonic"],
            settings["autotune_scale"],
            settings["autotune_strength"],
            settings["stereo_sound"],
            settings["audio_upscaling"],
        ],
        outputs=[synth_voice, converted_synth_voice],
    )
