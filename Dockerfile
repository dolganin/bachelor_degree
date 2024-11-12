FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    openjdk-11-jre-headless \
    python3.10 python3.10-venv python3.10-dev python3-pip \
    curl git cmake build-essential \
    libboost-all-dev \
    libsdl2-dev \
    libsndfile1-dev \
    libfftw3-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV KAFKA_HOME=/opt/kafka
RUN curl -O https://downloads.apache.org/kafka/3.8.1/kafka-3.8.1-src.tgz && \
    tar -xzf kafka-3.8.1-src.tgz && mv kafka-3.8.1-src $KAFKA_HOME && \
    rm kafka-3.8.1-src.tgz

WORKDIR /workspace
ENV VENV_PATH=/workspace/dith
ENV PATH="$VENV_PATH/bin:$PATH"

COPY . /workspace/

RUN if [ ! -d "/workspace/DoomITH" ]; then \
    git clone https://github.com/dolganin/DoomITH.git /workspace/DoomITH; \
fi

RUN mkdir -p /workspace/DoomITH/build && cd /workspace/DoomITH/build && \
    cmake .. && \
    make -j$(nproc) && \
    pip install . && \
    cd ..

COPY requirements.txt /workspace/requirements.txt

RUN [ ! -d "$VENV_PATH" ] && python3.10 -m venv $VENV_PATH || echo "Venv already created"

RUN . $VENV_PATH/bin/activate && \
    pip install --upgrade pip && \
    pip install -r /workspace/requirements.txt

COPY host_dith.conf /workspace/host_dith.conf

RUN chmod +x /workspace/scripts/create_kafka_topic.sh

EXPOSE 5000 6006 9092 2181
