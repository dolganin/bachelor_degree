#!/bin/bash
# Скрипт для создания Kafka топика

echo "Creating Kafka topic 'doom_screen'..."
$KAFKA_HOME/bin/kafka-topics.sh --create \
    --topic doom_screen \
    --bootstrap-server kafka:9092 \
    --partitions 3 \
    --replication-factor 1
