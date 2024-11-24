# Build stage for DoomITH
FROM ubuntu:22.04 AS build-doomith

RUN apt-get update && apt-get install -y git cmake build-essential libboost-all-dev libsdl2-dev libsndfile1-dev libfftw3-dev

WORKDIR /workspace
RUN git clone https://github.com/dolganin/DoomITH.git /workspace/DoomITH
WORKDIR /workspace/DoomITH/build
RUN cmake .. && make -j$(nproc)

# Final stage
FROM ubuntu:22.04

# Install necessary packages
RUN apt-get update && apt-get install -y openjdk-11-jre-headless python3.10 python3.10-venv python3.10-dev python3-pip curl git

# Set up Kafka
ENV KAFKA_HOME=/opt/kafka
RUN curl -O https://downloads.apache.org/kafka/3.8.1/kafka-3.8.1-src.tgz && tar -xzf kafka-3.8.1-src.tgz && mv kafka-3.8.1-src $KAFKA_HOME && rm kafka-3.8.1-src.tgz

# Set up virtual environment
WORKDIR /workspace
ENV VENV_PATH=/workspace/dith
ENV PATH="$VENV_PATH/bin:$PATH"
RUN python3.10 -m venv $VENV_PATH
RUN . $VENV_PATH/bin/activate && pip install --upgrade pip

# Copy built DoomITH package from build stage
COPY --from=build-doomith /workspace/DoomITH/build /workspace/DoomITH/build
RUN . $VENV_PATH/bin/activate && pip install /workspace/DoomITH

# Copy project code
COPY . /workspace/

# Copy additional configurations
COPY host_dith.conf /workspace/host_dith.conf
COPY scripts/create_kafka_topic.sh /workspace/scripts/create_kafka_topic.sh
RUN chmod +x /workspace/scripts/create_kafka_topic.sh

# Expose ports
EXPOSE 5000 6006 9092 2181