#!/bin/bash
# Запуск MuXVS (Linux / macOS). Требуется ./run-MuXVS-installer.sh
set -euo pipefail

cd "$(dirname "$0")"

ENV_PYTHON="env/bin/python"

if [ ! -x "$ENV_PYTHON" ]; then
    echo "Ошибка: окружение не найдено."
    echo "Сначала выполните './run-MuXVS-installer.sh'."
    exit 1
fi

check_internet_connection() {
    echo "Проверка подключения к интернету..."
    if curl -s --max-time 5 https://huggingface.co > /dev/null 2>&1; then
        echo "Подключение к интернету есть."
        INTERNET_AVAILABLE=1
    else
        echo "Подключение к интернету не обнаружено."
        INTERNET_AVAILABLE=0
    fi
    echo
}

running_interface() {
    echo "Запуск интерфейса..."
    if [ "$INTERNET_AVAILABLE" -eq 1 ]; then
        echo "Режим ONLINE..."
        exec "$ENV_PYTHON" app.py "$@"
    else
        echo "Режим OFFLINE..."
        exec "$ENV_PYTHON" app.py --offline "$@"
    fi
}

check_internet_connection
running_interface "$@"
