from confluent_kafka import Producer

# Конфигурация продюсера Kafka
producer_config = {
    'bootstrap.servers': '192.168.3.2:9092',
}

producer = Producer(producer_config)

def delivery_report(err, msg):
    """
    Обратный вызов для отчета о доставке сообщения.
    """
    if err is not None:
        print(f"Message delivery failed: {err}")
    else:
        pass
        # print(f"Message delivered to {msg.topic()} [{msg.partition()}]")

def publish_numpy_array(array):
    """
    Функция для публикации NumPy массива в Kafka.
    """
    try:
        # Преобразование массива в байты
        array_bytes = array.tobytes()  # Преобразуем NumPy массив в байты
        producer.produce('doom_screen', value=array_bytes, callback=delivery_report)  # Публикуем массив в топик
        producer.flush()  # Ждём доставки сообщений
    except Exception as e:
        print(f"Error publishing message: {e}")
