"""Общие константы и компоненты интерфейса MuXVS."""

import os
import re

import gradio as gr

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RVC_MODELS_DIR = os.path.join(os.getcwd(), "models", "RVC_models")
RVC_OUTPUT_DIR = os.path.join(os.getcwd(), "output", "RVC_output")
UVR_MODELS_DIR = os.path.join(os.getcwd(), "models", "UVR_models")
UVR_OUTPUT_DIR = os.path.join(os.getcwd(), "output", "UVR_output")

OUTPUT_FORMATS = ["wav", "flac", "mp3", "ogg", "m4a"]
F0_METHODS = ["hpa-rmvpe", "rmvpe+", "rmvpe", "fcpe", "crepe", "crepe-tiny"]

AUTOTUNE_NOTES = ["C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B"]
AUTOTUNE_SCALES = [
    "chromatic",
    "major",
    "minor",
    "dorian",
    "phrygian",
    "lydian",
    "mixolydian",
    "harmonic_minor",
    "melodic_minor",
    "pentatonic_major",
    "pentatonic_minor",
    "blues",
]

EDGE_VOICES = {
    "Английский (Великобритания)": ["en-GB-SoniaNeural", "en-GB-RyanNeural"],
    "Английский (США)": ["en-US-JennyNeural", "en-US-GuyNeural"],
    "Арабский (Египет)": ["ar-EG-SalmaNeural", "ar-EG-ShakirNeural"],
    "Арабский (Саудовская Аравия)": ["ar-SA-HamedNeural", "ar-SA-ZariyahNeural"],
    "Бенгальский (Бангладеш)": ["bn-BD-RubaiyatNeural", "bn-BD-KajalNeural"],
    "Венгерский": ["hu-HU-TamasNeural", "hu-HU-NoemiNeural"],
    "Вьетнамский": ["vi-VN-HoaiMyNeural", "vi-VN-HuongNeural"],
    "Греческий": ["el-GR-AthinaNeural", "el-GR-NestorasNeural"],
    "Датский": ["da-DK-PernilleNeural", "da-DK-MadsNeural"],
    "Иврит": ["he-IL-AvriNeural", "he-IL-HilaNeural"],
    "Испанский (Испания)": ["es-ES-ElviraNeural", "es-ES-AlvaroNeural"],
    "Испанский (Мексика)": ["es-MX-DaliaNeural", "es-MX-JorgeNeural"],
    "Итальянский": ["it-IT-ElsaNeural", "it-IT-DiegoNeural"],
    "Китайский (упрощенный)": ["zh-CN-XiaoxiaoNeural", "zh-CN-YunxiNeural"],
    "Корейский": ["ko-KR-SunHiNeural", "ko-KR-InJoonNeural"],
    "Немецкий": ["de-DE-KatjaNeural", "de-DE-ConradNeural"],
    "Нидерландский": ["nl-NL-ColetteNeural", "nl-NL-FennaNeural"],
    "Норвежский": ["nb-NO-PernilleNeural", "nb-NO-FinnNeural"],
    "Польский": ["pl-PL-MajaNeural", "pl-PL-JacekNeural"],
    "Португальский (Бразилия)": ["pt-BR-FranciscaNeural", "pt-BR-AntonioNeural"],
    "Португальский (Португалия)": ["pt-PT-RaquelNeural", "pt-PT-DuarteNeural"],
    "Румынский": ["ro-RO-EmilNeural", "ro-RO-AndreiNeural"],
    "Русский": ["ru-RU-SvetlanaNeural", "ru-RU-DmitryNeural"],
    "Тагальский": ["tl-PH-AngeloNeural", "tl-PH-TessaNeural"],
    "Тамильский": ["ta-IN-ValluvarNeural", "ta-IN-KannanNeural"],
    "Тайский": ["th-TH-PremwadeeNeural", "th-TH-NiwatNeural"],
    "Турецкий": ["tr-TR-AhmetNeural", "tr-TR-EmelNeural"],
    "Украинский": ["uk-UA-OstapNeural", "uk-UA-PolinaNeural"],
    "Филиппинский": ["fil-PH-AngeloNeural", "fil-PH-TessaNeural"],
    "Финский": ["fi-FI-NooraNeural", "fi-FI-SelmaNeural"],
    "Французский (Канада)": ["fr-CA-SylvieNeural", "fr-CA-AntoineNeural"],
    "Французский (Франция)": ["fr-FR-DeniseNeural", "fr-FR-HenriNeural"],
    "Чешский": ["cs-CZ-VlastaNeural", "cs-CZ-AntoninNeural"],
    "Шведский": ["sv-SE-HilleviNeural", "sv-SE-MattiasNeural"],
    "Японский": ["ja-JP-NanamiNeural", "ja-JP-KeitaNeural"],
}


def natural_key(text: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def list_rvc_models() -> list:
    """Имена папок голосовых моделей, отсортированные естественным порядком."""
    if not os.path.isdir(RVC_MODELS_DIR):
        return []
    return sorted(
        (name for name in os.listdir(RVC_MODELS_DIR) if os.path.isdir(os.path.join(RVC_MODELS_DIR, name))),
        key=natural_key,
    )


def refresh_models() -> gr.update:
    return gr.update(choices=list_rvc_models())


def model_select():
    """Группа выбора модели: список + кнопка обновления (классический вид)."""
    with gr.Group():
        model = gr.Dropdown(label="Голосовые модели:", choices=list_rvc_models())
        refresh_btn = gr.Button("Обновить список моделей", variant="primary")
    refresh_btn.click(refresh_models, outputs=model, api_name=False)
    return model


def pitch_group():
    """Группа тона: авто-тон + порог + слайдер (классический вид)."""
    with gr.Group():
        autopitch = gr.Checkbox(value=False, label="Автоматическое определение высоты тона")
        autopitch_threshold = gr.Radio(
            value=155.0,
            choices=[("Мужская модель", 155.0), ("Женская модель", 255.0)],
            show_label=False,
            visible=False,
        )
        rvc_pitch = gr.Slider(
            minimum=-24,
            maximum=24,
            step=1,
            value=0,
            label="Регулировка высоты тона",
            info="-24 — Мужская модель | 24 — Женская модель",
        )
    autopitch.change(_toggle_autopitch, inputs=autopitch, outputs=[autopitch_threshold, rvc_pitch], api_name=False)
    return autopitch, autopitch_threshold, rvc_pitch


def process_file_upload(file):
    return file, gr.update(value=file)


def swap_visibility():
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(value=""),
        gr.update(value=None),
    )


def swap_buttons():
    return gr.update(visible=False), gr.update(visible=True)


def update_edge_voices(language: str) -> gr.update:
    voices = EDGE_VOICES.get(language, [])
    return gr.update(choices=voices, value=voices[0] if voices else None)


def _toggle_autopitch(enabled: bool):
    if enabled:
        return gr.update(visible=True), gr.update(visible=False)
    return gr.update(visible=False), gr.update(visible=True)


def _toggle_autotune(enabled: bool):
    return gr.update(visible=enabled), gr.update(visible=enabled), gr.update(visible=enabled)


def conversion_settings():
    """Аккордеон тонких настроек конвертации (классический вид).

    Возвращает словарь компонентов; ключи соответствуют именованным
    аргументам `rvc_infer` / `rvc_batch_infer` / `rvc_edgetts_infer`
    (кроме модели, входа, тона и формата — они лежат в основной части).
    """
    with gr.Accordion("Настройки преобразования", open=False):
        with gr.Column(variant="panel"):
            with gr.Accordion("Стандартные настройки", open=False):
                with gr.Group():
                    with gr.Column(variant="panel"):
                        f0_method = gr.Dropdown(
                            value="rmvpe",
                            label="Метод выделения тона",
                            choices=F0_METHODS,
                        )
                    with gr.Column(variant="panel"):
                        index_rate = gr.Slider(
                            minimum=0,
                            maximum=1,
                            step=0.01,
                            value=0,
                            label="Влияние индекса",
                            info="Влияние, оказываемое индексным файлом; Чем выше значение, тем больше влияние. Однако выбор более низких значений может помочь смягчить артефакты, присутствующие в аудио.",
                        )
                        volume_envelope = gr.Slider(
                            minimum=0,
                            maximum=1,
                            step=0.01,
                            value=1,
                            label="Скорость смешивания RMS",
                            info="Заменить или смешать с огибающей громкости выходного сигнала. Чем ближе значение к 1, тем больше используется огибающая выходного сигнала.",
                        )
                        protect = gr.Slider(
                            minimum=0,
                            maximum=0.5,
                            step=0.01,
                            value=0.5,
                            label="Защита согласных",
                            info="Защитить согласные и звуки дыхания, чтобы избежать электроакустических разрывов и артефактов. Максимальное значение параметра 0.5 обеспечивает полную защиту. Уменьшение этого значения может снизить защиту, но уменьшить эффект индексирования.",
                        )

            with gr.Accordion("Дополнительные настройки", open=False):
                with gr.Group():
                    with gr.Column():
                        with gr.Row(variant="panel"):
                            stereo_sound = gr.Checkbox(
                                value=False,
                                label="Преобразовать в стерео",
                                info="Преобразование моно звука в стерео",
                            )
                            audio_upscaling = gr.Checkbox(
                                value=False,
                                label="Аудио-апскейл",
                                info="Улучшение качества аудио (долгая обработка)",
                            )
                            with gr.Column():
                                autotune = gr.Checkbox(
                                    value=False,
                                    label="АвтоТюн",
                                    info="Коррекция высоты тона",
                                )
                                with gr.Column():
                                    with gr.Row():
                                        autotune_tonic = gr.Dropdown(
                                            value="C",
                                            label="Тоника",
                                            choices=AUTOTUNE_NOTES,
                                            visible=False,
                                        )
                                        autotune_scale = gr.Dropdown(
                                            value="chromatic",
                                            label="Гамма/Лад",
                                            choices=AUTOTUNE_SCALES,
                                            visible=False,
                                        )
                                    autotune_strength = gr.Slider(
                                        minimum=0,
                                        maximum=1,
                                        step=0.1,
                                        value=1,
                                        label="Сила коррекции",
                                        visible=False,
                                    )
                        with gr.Row(variant="panel"):
                            f0_min = gr.Slider(
                                minimum=1,
                                maximum=120,
                                step=1,
                                value=50,
                                label="Минимальный диапазон тона",
                                info="Определяет нижнюю границу диапазона тона, который алгоритм будет использовать для определения основной частоты (F0) в аудиосигнале.",
                            )
                            f0_max = gr.Slider(
                                minimum=380,
                                maximum=16000,
                                step=1,
                                value=1100,
                                label="Максимальный диапазон тона",
                                info="Определяет верхнюю границу диапазона тона, который алгоритм будет использовать для определения основной частоты (F0) в аудиосигнале.",
                            )

    autotune.change(
        _toggle_autotune,
        inputs=autotune,
        outputs=[autotune_tonic, autotune_scale, autotune_strength],
        api_name=False,
    )

    return {
        "f0_method": f0_method,
        "index_rate": index_rate,
        "protect": protect,
        "volume_envelope": volume_envelope,
        "f0_min": f0_min,
        "f0_max": f0_max,
        "stereo_sound": stereo_sound,
        "audio_upscaling": audio_upscaling,
        "autotune": autotune,
        "autotune_tonic": autotune_tonic,
        "autotune_scale": autotune_scale,
        "autotune_strength": autotune_strength,
    }
