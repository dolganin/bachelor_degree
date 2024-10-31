#!/bin/bash
# Скрипт для запуска Kafka и всех скриптов

# Запуск Zookeeper и Kafka
$KAFKA_HOME/bin/zookeeper-server-start.sh -daemon $KAFKA_HOME/config/zookeeper.properties
$KAFKA_HOME/bin/kafka-server-start.sh -daemon $KAFKA_HOME/config/server.properties

# Ожидание запуска Kafka
sleep 5

# Создание топика
/workspace/create_kafka_topic.sh

# Активация виртуального окружения
source $VENV_PATH/bin/activate

# Запуск server_consumer/kafka_consumer.py, server_consumer/flask_server.py и main.py
python /workspace/server_consumer/kafka_consumer.py &
python /workspace/server_consumer/flask_server.py &
python /workspace/main.py "$@" &

# Ожидание завершения процессов
wait
