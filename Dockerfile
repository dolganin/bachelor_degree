FROM nvidia/cuda:11.7.1-base-ubuntu22.04

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gdb \
    xvfb \
    x11-utils \
    libglew-dev \
    libxi-dev \
    openjdk-11-jre-headless \
    python3.10 python3.10-venv python3.10-dev python3-pip \
    curl git cmake build-essential \
    libboost-all-dev \
    libsdl2-dev \
    libsndfile1-dev \
    libfftw3-dev \
    libgl1-mesa-glx \
    libegl1-mesa \
    libglu1-mesa-dev freeglut3-dev mesa-common-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Установка Kafka
ENV KAFKA_HOME=/opt/kafka
RUN curl -O https://downloads.apache.org/kafka/3.8.1/kafka-3.8.1-src.tgz && \
    tar -xzf kafka-3.8.1-src.tgz && mv kafka-3.8.1-src $KAFKA_HOME && \
    rm kafka-3.8.1-src.tgz

# Настройка рабочей директории
WORKDIR /workspace

# Создание и активация виртуального окружения Python
ENV VENV_PATH=/workspace/venv
ENV PATH="$VENV_PATH/bin:$PATH"
RUN python3.10 -m venv $VENV_PATH
RUN . $VENV_PATH/bin/activate && \
    pip install --upgrade pip && \
    pip install numpy

# Копирование зависимостей Python для frontend
COPY reqs/front_requirements.txt /workspace/front_requirements.txt
RUN . $VENV_PATH/bin/activate && pip install -r /workspace/front_requirements.txt

# Копирование только необходимых файлов и директорий
COPY server_consumer /workspace/server_consumer
COPY .env /workspace/.env
COPY host_dith.conf /workspace/host_dith.conf

# Установка Kafka consumer/server как команды по умолчанию
CMD ["tail", "-f", "/dev/null"]