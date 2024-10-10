import json
import numpy as np
from confluent_kafka import Producer
import base64
import cv2

# Конфигурация продюсера Kafka
producer_config = {
    'bootstrap.servers': '192.168.3.2:9092',
}

producer = Producer(producer_config)

def delivery_report(err, msg):
    """Обратный вызов для отчета о доставке сообщения."""
    if err is not None:
        print(f"Message delivery failed: {err}")
    else:
        pass

def publish_data(array: np.ndarray, epoch: int, loss: float, mode: str, mean_reward: float):
    """Публикация данных (NumPy массив и метаданные) в Kafka."""
    try:
        # Преобразуем NumPy массив в байты и затем в base64 строку
        _, buffer = cv2.imencode('.jpg', array)
        array_base64 = base64.b64encode(buffer).decode('utf-8')

        loss = float(loss)
        mean_reward = float(mean_reward)
        
        # Формируем структуру сообщения
        message_data = {
            'image': array_base64,
            'epoch': epoch,
            'loss': loss,
            'mode': mode,
            'meanReward': mean_reward
        }
        
        # Преобразуем структуру в JSON строку
        message_json = json.dumps(message_data)
        
        # Публикуем сообщение в топик Kafka
        producer.produce('doom_screen', value=message_json, callback=delivery_report)
        producer.flush()
    except Exception as e:
        print(f"Error publishing message: {e}")

