# Базовый образ Ubuntu 22.04
FROM ubuntu:22.04

# Устанавливаем обновления и основные зависимости
RUN apt-get update && apt-get install -y \
    openjdk-11-jre-headless \
    python3.10 python3.10-venv python3.10-dev python3-pip \
    curl git cmake build-essential \
    libboost-all-dev \
    libsdl2-dev \
    libsndfile1-dev \
    libfftw3-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Установка Kafka
ENV KAFKA_HOME=/opt/kafka
RUN curl -O https://downloads.apache.org/kafka/3.8.1/kafka-3.8.1-src.tgz && \
    tar -xzf kafka-3.8.1-src.tgz && mv kafka-3.8.1-src $KAFKA_HOME && \
    rm kafka-3.8.1-src.tgz

# Создаем директории и переменные среды
WORKDIR /workspace
ENV VENV_PATH=/workspace/dith
ENV PATH="$VENV_PATH/bin:$PATH"

# Копируем файлы репозитория проекта и DoomITH
COPY . /workspace/

# Клонируем репозиторий DoomITH (если его нет)
RUN if [ ! -d "/workspace/DoomITH" ]; then \
    git clone https://github.com/dolganin/DoomITH.git /workspace/DoomITH; \
fi

# Создаем папку для сборки и собираем проект DoomITH, если необходимо
RUN mkdir -p /workspace/DoomITH/build && cd /workspace/DoomITH/build && \
    cmake .. && \
    make -j$(nproc) && \
    pip install . && \
    cd ..

# Копируем файл requirements.txt в контейнер и устанавливаем зависимости
COPY requirements.txt /workspace/requirements.txt

# Если виртуальное окружение еще не создано, создаем его, иначе пропускаем
RUN [ ! -d "$VENV_PATH" ] && python3.10 -m venv $VENV_PATH || echo "Venv already created"

# Устанавливаем зависимости
RUN . $VENV_PATH/bin/activate && \
    pip install --upgrade pip && \
    pip install -r /workspace/requirements.txt

# Копируем конфигурационный файл для Kafka (если требуется для удаленного режима)
COPY host_dith.conf /workspace/host_dith.conf

# Копируем bash скрипты для автоматизации Kafka и запуска приложения
RUN chmod +x /workspace/scripts/create_kafka_topic.sh

# Открываем порты для Kafka, Flask и TensorBoard
EXPOSE 5000 6006 9092 2181

# Нет команды CMD здесь — запуск будет определяться через docker-compose.yml
