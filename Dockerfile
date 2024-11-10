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

# Клонируем репозиторий проекта (основной проект)
COPY . /workspace/

# Клонируем репозиторий DoomITH
RUN git clone https://github.com/dolganin/DoomITH.git /workspace/DoomITH

# Создаем папку для сборки и переходим в неё, параллельно строим проект на всех ядрах
RUN mkdir /workspace/DoomITH/build && cd /workspace/DoomITH/build && \
    cmake .. && \
    make -j$(nproc) && \ 
    cd .. && pip install . && \
    cd ..

# Копируем файл requirements.txt из текущей директории на хосте в контейнер
COPY requirements.txt /workspace/requirements.txt

# Создаем и активируем виртуальное окружение
RUN python3.10 -m venv $VENV_PATH && \
    . $VENV_PATH/bin/activate && \
    pip install --upgrade pip && \
    pip install -r /workspace/requirements.txt

# Копируем конфигурационный файл для хоста (host_dith.conf) в контейнер
COPY host_dith.conf /workspace/host_dith.conf

# Копируем bash скрипты для автоматизации Kafka и запуска приложения
RUN chmod +x /workspace/create_kafka_topic.sh /workspace/start_services.sh

# Открываем необходимые порты
EXPOSE 5000 6006

# Команда запуска приложения
CMD ["/workspace/scripts/start_services.sh"]
