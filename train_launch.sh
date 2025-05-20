#!/bin/bash

# Get the script's directory
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

# Define the name of the virtual environment directory
VENV_DIR="$SCRIPT_DIR/bd_env"

# Check if the virtual environment directory exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Создание виртуального окружения в $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

# Check if reqs/nn_requirements.txt exists
REQ_FILE="$SCRIPT_DIR/reqs/nn_requirements.txt"
if [ ! -f "$REQ_FILE" ]; then
    echo "Ошибка: не найден $REQ_FILE."
    deactivate
    exit 1
fi

# Calculate the current hash of nn_requirements.txt
CURRENT_HASH=$(md5sum "$REQ_FILE" | awk '{print $1}')

# Check the stored hash
STORED_HASH_FILE="$VENV_DIR/.requirements_hash"
if [ -f "$STORED_HASH_FILE" ]; then
    STORED_HASH=$(cat "$STORED_HASH_FILE")
else
    STORED_HASH=""
fi

# Compare hashes
if [ "$CURRENT_HASH" != "$STORED_HASH" ]; then
    echo "Установка зависимостей из $REQ_FILE..."
    pip install --upgrade pip
    pip install -r "$REQ_FILE" || { echo "Ошибка при установке зависимостей."; deactivate; exit 1; }
    echo "$CURRENT_HASH" > "$STORED_HASH_FILE"
else
    echo "Зависимости уже установлены и не изменились."
fi

# DoomITH
DOOMITH_DIR="$SCRIPT_DIR/DoomITH"
if [ ! -d "$DOOMITH_DIR/.git" ]; then
    echo "Клонируем DoomITH..."
    git clone https://github.com/dolganin/DoomITH.git "$DOOMITH_DIR"
    echo "Собираем и устанавливаем DoomITH..."
    cd "$DOOMITH_DIR" || exit
    mkdir -p build
    cd build || exit
    cmake ..
    make -j$(nproc)
    cd ..
    pip install .
    cd "$SCRIPT_DIR"
fi

# coNNquest
CONNQUEST_DIR="$SCRIPT_DIR/coNNquest"
if [ ! -d "$CONNQUEST_DIR/.git" ]; then
    echo "Клонируем coNNquest..."
    git clone https://github.com/dolganin/coNNquest.git "$CONNQUEST_DIR"
    pip install -e "$CONNQUEST_DIR"
fi

# Create weights directory if missing
WEIGHTS_DIR="$SCRIPT_DIR/weights"
if [ ! -d "$WEIGHTS_DIR" ]; then
    echo "Создаём директорию весов: $WEIGHTS_DIR..."
    mkdir -p "$WEIGHTS_DIR"
fi

# Run the training script
echo "Запуск обучения..."
python ppo_main.py "$@"

# Deactivate venv
deactivate
