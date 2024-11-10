#!/bin/bash

# Путь к виртуальной среде
VENV_PATH="/workspace/python_envs/bachelor_degree/bin/activate"

# Проверяем, существует ли виртуальная среда
if [ ! -f "$VENV_PATH" ]; then
    echo "Виртуальная среда не найдена по пути $VENV_PATH"
    exit 1
fi

# Активируем виртуальную среду
source $VENV_PATH

# Запускаем main.py с передачей всех аргументов скрипту
python main.py "$@"