#!/bin/bash

# Проверяем, установлена ли переменная окружения BDEGREE
if [ -z "$BDEGREE" ]; then
  echo "Ошибка: Переменная окружения BDEGREE не установлена."
  exit 1
fi

# Активируем виртуальное окружение
source "$BDEGREE"

# Проверяем, был ли скрипт main.py указан и существует ли он
if [ ! -f "main.py" ]; then
  echo "Ошибка: Файл main.py не найден."
  exit 1
fi

# Запускаем скрипт main.py
python main.py

