import numpy as np
import json
from confluent_kafka import Consumer, KafkaError
import logging
import requests
import time

# Настройка логирования
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def send_frame_to_server(image_base64, epoch, loss, mode, mean_reward):
    """Отправка кадра и метаданных на Flask сервер по HTTP."""
    try:
        # Отправляем POST-запрос на сервер
        response = requests.post(
            "http://localhost:5000/update_frame", 
            json={
                'image': image_base64, 
                'epoch': epoch, 
                'loss': loss, 
                'mode': mode, 
                'meanReward': mean_reward
            }
        )
        if response.status_code == 200:
            logger.info("Frame sent successfully.")
        else:
            logger.error(f"Failed to send frame: {response.status_code} {response.text}")
    except Exception as e:
        logger.error(f"Error sending frame: {e}")

def consume_kafka_messages():
    """Потребление сообщений из Kafka топика 'doom_screen'."""
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

            # Декодируем JSON сообщение
            try:
                message_data = json.loads(msg.value().decode('utf-8'))
                image_base64 = message_data.get('image')
                epoch = message_data.get('epoch', 'Undefined')
                loss = message_data.get('loss', 'NaN')
                mode = message_data.get('mode', 'Unknown')
                mean_reward = message_data.get('meanReward', 'NaN')

                # Отправляем изображение и метаданные на Flask сервер
                send_frame_to_server(image_base64, epoch, loss, mode, mean_reward)
                logger.info("Frame and metadata sent successfully.")
            except Exception as decode_error:
                logger.error(f"Error decoding message: {decode_error}")

        except Exception as e:
            logger.error(f"Error in consume_kafka_messages: {e}")
            logger.info("Retrying connection to Kafka in 5 seconds...")
            time.sleep(5)

    consumer.close()
    logger.info("Kafka consumer closed.")

if __name__ == '__main__':
    consume_kafka_messages()
