import os
import socket
import time
import json
import logging
import requests
import base64
import io
from confluent_kafka import Consumer, KafkaError
from PIL import Image, ImageDraw, ImageFont

import numpy as np

# Настройка логирования
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def create_black_image_with_text(text):
    """Создание черного изображения с заданным текстом."""
    img = Image.new('RGB', (1280, 720), color=(0, 0, 0))
    d = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("static/fonts/amazdoomleft.ttf", 52)
    except IOError:
        font = ImageFont.load_default()

    text_bbox = d.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    x = (img.width - text_width) // 2
    y = (img.height - text_height) // 2

    d.text((x, y), text, fill=(255, 255, 255), font=font)

    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def get_server_url():
    """Получение URL сервера из конфигурационного файла host_dith.conf."""
    mode = os.getenv('MODE', 'local')
    if mode == 'remote':
        try:
            with open('host_dith.conf', 'r') as file:
                host_config = file.read().strip()
            logger.info(f"Remote server URL: {host_config}")
            return host_config
        except Exception as e:
            logger.error(f"Failed to read host_dith.conf: {e}")
            return None
    return "http://localhost:5000/update_frame"

def get_hostname():
    """Получение имени хоста для добавления в JSON данные."""
    return socket.gethostname()

def send_frame_to_server(image_base64, epoch, loss, mode, mean_reward, server_url):
    """Отправка кадра и метаданных на сервер по HTTP."""
    hostname = get_hostname()
    data = {
        'image': image_base64,
        'epoch': epoch,
        'loss': loss,
        'mode': mode,
        'meanReward': mean_reward,
        'hostname': hostname
    }

    try:
        response = requests.post(server_url, json=data)
        if response.status_code == 200:
            logger.info("Frame sent successfully.")
        else:
            logger.error(f"Failed to send frame: {response.status_code} {response.text}")
    except Exception as e:
        logger.error(f"Error sending frame: {e}")

def validate_and_process_message(message_data, server_url):
    """Валидация и обработка сообщения перед отправкой."""
    image_base64 = message_data.get('image')
    epoch = message_data.get('epoch', 'Undefined')
    loss = message_data.get('loss', 'NaN')
    mode = message_data.get('mode', 'Unknown')
    mean_reward = message_data.get('meanReward', 'NaN')

    if not image_base64:
        logger.info("Received empty image. Using default black image.")
        image_base64 = create_black_image_with_text("DITH isn't learning \n right now")
    else:
        try:
            image_data = base64.b64decode(image_base64)
            img = Image.open(io.BytesIO(image_data))
            img.verify()
        except Exception as img_error:
            logger.warning("Invalid image. Using default black image.")
            image_base64 = create_black_image_with_text("DITH isn't learning \n right now")

    send_frame_to_server(image_base64, epoch, loss, mode, mean_reward, server_url)

def consume_kafka_messages():
    """Потребление сообщений из Kafka топика 'doom_screen'."""
    consumer_config = {
        'bootstrap.servers': '0.0.0.0:9092',
        'group.id': 'flask-consumer-group',
        'auto.offset.reset': 'earliest'
    }

    consumer = Consumer(consumer_config)
    consumer.subscribe(['doom_screen'])
    logger.info("Subscribed to 'doom_screen'")

    server_url = get_server_url()
    if server_url is None:
        logger.error("No valid server URL found. Exiting...")
        return

    last_message_time = time.time()

    while True:
        try:
            msg = consumer.poll(timeout=0.5)
            current_time = time.time()

            if current_time - last_message_time > 2:
                logger.info("No messages for 2 seconds. Sending default image.")
                default_image = create_black_image_with_text("DITH isn't learning \n right now")
                send_frame_to_server(default_image, 'Undefined', 'NaN', 'Undefined', 'NaN', server_url)
                last_message_time = current_time

            if msg is None:
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    logger.error(f"Kafka Error: {msg.error()}")
                    break

            try:
                message_data = json.loads(msg.value().decode('utf-8'))
                validate_and_process_message(message_data, server_url)
                last_message_time = time.time()
                logger.info("Frame and metadata processed successfully.")
            except json.JSONDecodeError as decode_error:
                logger.error(f"JSON decode error: {decode_error}")
            except Exception as decode_error:
                logger.error(f"Message processing error: {decode_error}")

        except Exception as e:
            logger.error(f"Kafka consumption error: {e}")
            logger.info("Retrying connection to Kafka in 5 seconds...")
            time.sleep(5)

    consumer.close()
    logger.info("Kafka consumer closed.")

if __name__ == '__main__':
    consume_kafka_messages()
