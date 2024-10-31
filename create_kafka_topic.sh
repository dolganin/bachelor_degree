#!/bin/bash
# Скрипт для создания Kafka топика

$KAFKA_HOME/bin/kafka-topics.sh --create \
    --topic doom_screen \
    --bootstrap-server localhost:9092 \
    --partitions 3 \
    --replication-factor 1
