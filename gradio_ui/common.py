"""Общие константы и компоненты интерфейса MuXVS."""

import os
import re

import gradio as gr

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RVC_MODELS_DIR = os.path.join(os.getcwd(), "models", "RVC_models")
RVC_OUTPUT_DIR = os.path.join(os.getcwd(), "output", "RVC_output")
UVR_MODELS_DIR = os.path.join(os.getcwd(), "models", "UVR_models")
UVR_OUTPUT_DIR = os.path.join(os.getcwd(), "output", "UVR_output")
LOGS_DIR = os.path.join(os.getcwd(), "logs")

OUTPUT_FORMATS = ["wav", "flac", "mp3", "ogg", "m4a"]
F0_METHODS = ["rmvpe", "rmvpe+", "hpa-rmvpe", "fcpe", "crepe", "crepe-tiny"]

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


def model_row():
    """Строка выбора модели: выпадающий список + кнопка обновления."""
    with gr.Row():
        model = gr.Dropdown(label="Голосовая модель", choices=list_rvc_models(), scale=4)
        refresh_btn = gr.Button("⟳ Обновить", scale=1)
    refresh_btn.click(refresh_models, outputs=model, api_name=False)
    return model


def update_edge_voices(language: str) -> gr.update:
    voices = EDGE_VOICES.get(language, [])
    return gr.update(choices=voices, value=voices[0] if voices else None)


def _toggle_autopitch(enabled: bool):
    if enabled:
        return gr.update(visible=True), gr.update(visible=False)
    return gr.update(visible=False), gr.update(visible=True)


def _toggle_autotune(enabled: bool):
    return gr.update(visible=enabled), gr.update(visible=enabled), gr.update(visible=enabled)


def conversion_settings(pitch=None):
    """Аккордеон тонких настроек конвертации. Возвращает словарь компонентов.

    Ключи соответствуют именованным аргументам `rvc_infer` / `rvc_batch_infer`
    (кроме модели, входа, тона, метода F0 и формата — они лежат в основной части).
    Если передан слайдер тона, авто-тон скрывает его и показывает порог вместо него.
    """
    with gr.Accordion("Настройки конвертации", open=False):
        with gr.Row():
            index_rate = gr.Slider(minimum=0, maximum=1, step=0.01, value=0, label="Влияние индекса")
            protect = gr.Slider(minimum=0, maximum=0.5, step=0.01, value=0.5, label="Защита согласных")
            volume_envelope = gr.Slider(minimum=0, maximum=1, step=0.01, value=1, label="Смешивание громкости")
        with gr.Row():
            f0_min = gr.Slider(minimum=1, maximum=120, step=1, value=50, label="Мин. тон")
            f0_max = gr.Slider(minimum=380, maximum=16000, step=1, value=1100, label="Макс. тон")
            stereo_sound = gr.Checkbox(False, label="Стерео")
            audio_upscaling = gr.Checkbox(False, label="Апскейл (FlashSR)")
        with gr.Row():
            autopitch = gr.Checkbox(False, label="Авто-тон")
            autotune = gr.Checkbox(False, label="АвтоТюн")
        with gr.Row():
            autopitch_threshold = gr.Radio(
                [("Мужская модель", 155.0), ("Женская модель", 255.0)],
                value=155.0,
                show_label=False,
                visible=False,
            )
        with gr.Row():
            autotune_tonic = gr.Dropdown(AUTOTUNE_NOTES, value="C", label="Тоника", visible=False)
            autotune_scale = gr.Dropdown(AUTOTUNE_SCALES, value="chromatic", label="Лад", visible=False)
            autotune_strength = gr.Slider(minimum=0, maximum=1, step=0.1, value=1, label="Сила автотюна", visible=False)

    if pitch is None:
        autopitch.change(
            lambda enabled: gr.update(visible=enabled),
            inputs=autopitch,
            outputs=autopitch_threshold,
            api_name=False,
        )
    else:
        autopitch.change(_toggle_autopitch, inputs=autopitch, outputs=[autopitch_threshold, pitch], api_name=False)
    autotune.change(
        _toggle_autotune,
        inputs=autotune,
        outputs=[autotune_tonic, autotune_scale, autotune_strength],
        api_name=False,
    )

    return {
        "index_rate": index_rate,
        "protect": protect,
        "volume_envelope": volume_envelope,
        "f0_min": f0_min,
        "f0_max": f0_max,
        "stereo_sound": stereo_sound,
        "audio_upscaling": audio_upscaling,
        "autopitch": autopitch,
        "autopitch_threshold": autopitch_threshold,
        "autotune": autotune,
        "autotune_tonic": autotune_tonic,
        "autotune_scale": autotune_scale,
        "autotune_strength": autotune_strength,
    }
