from confluent_kafka import Producer
import numpy as np
import time

# Конфигурация продюсера Kafka
producer_config = {
    'bootstrap.servers': 'localhost:9092',  # Адрес Kafka-брокера
}

producer = Producer(producer_config)

def delivery_report(err, msg):
    """
    Обратный вызов для отчета о доставке сообщения.
    """
    if err is not None:
        print(f"Message delivery failed: {err}")
    else:
        print(f"Message delivered to {msg.topic()} [{msg.partition()}]")

def publish_numpy_array(array):
    """
    Функция для публикации NumPy массива в Kafka.
    """
    # Преобразование массива в байты
    array_bytes = array.tobytes()  # Преобразуем NumPy массив в байты
    producer.produce('doom_screen', value=array_bytes, callback=delivery_report)  # Публикуем массив в топик
    producer.poll(0)  # Обрабатываем обратные вызовы
