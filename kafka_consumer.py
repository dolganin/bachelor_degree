# kafka_consumer.py

import base64
import cv2
import numpy as np
from confluent_kafka import Consumer, KafkaError
import logging
import queue
import time
import requests

# Настройка логирования
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Создание очереди для хранения кадров
frame_queue = queue.Queue(maxsize=1000)

def send_frame_to_server(frame):
    """Отправка кадра на Flask сервер по HTTP."""
    _, buffer = cv2.imencode('.jpg', frame)  # Кодируем в JPEG
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    # Отправляем POST-запрос на сервер
    try:
        response = requests.post("http://localhost:5000/update_frame", json={'image': encoded_image})
        if response.status_code == 200:
            logger.info("Frame sent successfully.")
        else:
            logger.error(f"Failed to send frame: {response.status_code} {response.text}")
    except Exception as e:
        logger.error(f"Error sending frame: {e}")

def consume_kafka_messages():
    """Функция для потребления сообщений из Kafka топика 'doom_screen'."""
    consumer_config = {
        'bootstrap.servers': '192.168.3.2:9092',
        'group.id': 'flask-consumer-group',
        'auto.offset.reset': 'earliest'
    }

    consumer = Consumer(consumer_config)
    consumer.subscribe(['doom_screen'])
    logger.info("Kafka consumer subscribed to 'doom_screen'")

    while True:
        try:
            msg = consumer.poll(timeout=0.33)
            if msg is None:
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    logger.error(f"Kafka Error: {msg.error()}")
                    break

            image_data = msg.value()
            logger.debug(f"Received message of size: {len(image_data)} bytes.")
            try:
                frame = np.frombuffer(image_data, np.uint8).reshape(480, 640, 3)
                if frame is not None:
                    send_frame_to_server(frame)
                    logger.info("Frame decoded and updated successfully.")
                else:
                    logger.warning("Failed to decode frame.")
            except Exception as decode_error:
                logger.error(f"Error decoding frame: {decode_error}")

        except Exception as e:
            logger.error(f"Error in consume_kafka_messages: {e}")
            logger.info("Retrying connection to Kafka in 5 seconds...")
            time.sleep(5)
    
    consumer.close()
    logger.info("Kafka consumer closed.")

if __name__ == '__main__':
    consume_kafka_messages()
