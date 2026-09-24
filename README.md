# MuXVS

Чуть другая реализация RVC: конвертация голоса, синтез речи, разделение аудио и обучение моделей в одном минималистичном веб-интерфейсе на Gradio.

## Вкладки

- **Конвертация** — замена голоса в одном файле, тонкие настройки, пакетная конвертация папки.
- **TTS** — синтез речи из текста (EdgeTTS) + замена голоса.
- **UVR** — разделение аудио на стемы (Roformer, MDX23C, MDX-NET, VR Arch, Demucs).
- **Обучение** — полный цикл как в официальном RVC: нарезка → признаки → обучение + индекс.
- **Модели** — загрузка RVC-моделей (ссылка, ZIP, файлы) и установка эмбеддеров.
- **Инструменты** — проверка моделей, разделение чекпоинта G/D, апскейл аудио (FlashSR).

## Запуск

```bash
./run-MuXVS-installer.sh      # один раз: окружение и зависимости
./run-MuXVS.sh                # интерфейс (сам выберет online/offline режим)
```

Windows: `run-MuXVS-installer.bat`, затем `run-MuXVS.bat`.

Установщик всё делает сам: Python и GPU/CPU выбираются автоматически, параметры не нужны.

Вручную: `python app.py [--port 4000] [--server-name 127.0.0.1] [--offline]`.

Запуск в Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MuXolotl/MuXVS/blob/main/assets/MuXVS_Colab.ipynb) — установка, проверка окружения и запуск интерфейса с выбором способа доступа.

## CLI

```bash
python -m rvc.inference.infer_cli rvc --rvc_model MyModel --input_path in.wav
python -m rvc.inference.infer_cli tts --rvc_model MyModel --tts_voice ru-RU-SvetlanaNeural --tts_text "Привет"
```

> **Python:** дистрибутивы собираются с Python **3.11**; CI дополнительно тестирует код на 3.10–3.13.
> Готовые сборки публикуются в HuggingFace-репозитории [Politrees/MuXVS](https://huggingface.co/Politrees/MuXVS).
