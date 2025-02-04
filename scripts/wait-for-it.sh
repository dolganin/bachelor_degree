#!/bin/bash
# wait-for-it.sh
host="$1"
shift
port="$1"
shift
timeout="$1"
shift

echo "Waiting for $host:$port..."

# Пытаемся подключиться к Kafka, пока не будет доступен
nc -z "$host" "$port"
status=$?
while [ $status -ne 0 ]; do
    sleep 1
    nc -z "$host" "$port"
    status=$?
done

echo "$host:$port is available"
exec "$@"
