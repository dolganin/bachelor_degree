import numpy as np
import json
from confluent_kafka import Consumer, KafkaError
import logging
import requests
import time
from PIL import Image, ImageDraw, ImageFont
import io
import base64

# Настройка логирования
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def create_black_image_with_text(text):
    """Создание черного изображения с заданным текстом."""
    # Создаем черное изображение размером 640x480
    img = Image.new('RGB', (640, 480), color=(0, 0, 0))
    d = ImageDraw.Draw(img)

    # Задаем шрифт и размер
    try:
        font = ImageFont.truetype("static/fonts/amazdoomleft.ttf", 52)  # Путь к шрифту Arial
    except IOError:
        font = ImageFont.load_default()  # Используем стандартный шрифт

    # Вычисляем размер текста для центрирования
    text_bbox = d.textbbox((0, 0), text, font=font)  # Получаем границы текста
    text_width = text_bbox[2] - text_bbox[0]  # ширина текста
    text_height = text_bbox[3] - text_bbox[1]  # высота текста
    x = (img.width - text_width) // 2
    y = (img.height - text_height) // 2

    # Рисуем текст на изображении
    d.text((x, y), text, fill=(255, 255, 255), font=font)
    
    # Преобразуем изображение в base64
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return img_base64

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
            msg = consumer.poll(timeout=0.5)
            if msg is None:
                logger.info("Received None message. Sending default black image.")
                image_base64 = create_black_image_with_text("DITH isn't learning \n right now")
                send_frame_to_server(image_base64, 'Undefined', 'NaN', 'Undefined', 'NaN')
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

                # Проверяем, если изображение пустое или некорректное
                if not image_base64:
                    logger.info("Received empty image. Sending default black image.")
                    image_base64 = create_black_image_with_text("DITH isn't learning \n right now")
                else:
                    # Дополнительно можно добавить проверку корректности изображения (например, базовую проверку на формат)
                    try:
                        # Пробуем декодировать базу64 в изображение
                        image_data = base64.b64decode(image_base64)
                        img = Image.open(io.BytesIO(image_data))
                        img.verify()  # Проверка, что изображение корректное
                    except Exception as img_error:
                        logger.warning("Received an invalid image. Sending default black image.")
                        image_base64 = create_black_image_with_text("DITH isn't learning \n right now")

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
