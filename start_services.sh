#!/bin/bash
# Скрипт для запуска Kafka и всех сервисов

# Проверяем режим работы (по умолчанию локальный режим)
MODE=${MODE:-local}

# Запуск Zookeeper и Kafka
$KAFKA_HOME/bin/zookeeper-server-start.sh -daemon $KAFKA_HOME/config/zookeeper.properties
$KAFKA_HOME/bin/kafka-server-start.sh -daemon $KAFKA_HOME/config/server.properties

# Ожидание запуска Kafka
sleep 5

# Создание топика
/workspace/create_kafka_topic.sh

# Активация виртуального окружения
source $VENV_PATH/bin/activate

# В зависимости от режима работы, запускаем соответствующие процессы
if [ "$MODE" == "remote" ]; then
  # В режиме удаленного сервера, передаем информацию на удаленный сервер
  echo "Running in remote mode. Sending data to remote server."
  python /workspace/server_consumer/kafka_consumer.py &
else
  # В локальном режиме, запускаем сервер и консюмера
  echo "Running in local mode. Running Flask server and Kafka consumer locally."
  python /workspace/server_consumer/kafka_consumer.py &
  python /workspace/server_consumer/flask_server.py &
fi

# Запуск основного скрипта
python /workspace/main.py "$@" &

# Запуск TensorBoard
tensorboard --logdir=/workspace/runs --port=6006 &

# Ожидание завершения процессов
wait
