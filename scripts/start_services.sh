#!/bin/bash
# Скрипт для запуска Kafka и всех сервисов

# Проверяем режим работы (по умолчанию локальный режим)
MODE=${MODE:-local}

# Запуск Zookeeper и Kafka (доступны в обоих режимах для консюмации данных)
$KAFKA_HOME/bin/zookeeper-server-start.sh -daemon $KAFKA_HOME/config/zookeeper.properties
$KAFKA_HOME/bin/kafka-server-start.sh -daemon $KAFKA_HOME/config/server.properties

# Ожидание запуска Kafka
sleep 5

# Создание топика, если он еще не существует
/workspace/create_kafka_topic.sh

# Активация виртуального окружения
source $VENV_PATH/bin/activate

if [ "$MODE" == "remote" ]; then
  # В удаленном режиме запустим Kafka-консюмер, без Flask-сервера
  echo "Running in remote mode. Kafka consumer will be launched only."
  python /workspace/server_consumer/kafka_consumer.py &
else
  # В локальном режиме запускаем Flask сервер и Kafka консюмер
  echo "Running in local mode. Flask server and Kafka consumer will be launched."
  python /workspace/server_consumer/kafka_consumer.py &  # запуск Kafka-консюмера
  python /workspace/server_consumer/flask_server.py &     # запуск Flask сервера
fi

# Запуск основного скрипта (передача аргументов, если необходимо)
python /workspace/main.py "$@" > /workspace/main.log 2>&1 &

# Запуск TensorBoard
tensorboard --logdir=/workspace/runs --port=6006 &

# Ожидание завершения процессов
wait
