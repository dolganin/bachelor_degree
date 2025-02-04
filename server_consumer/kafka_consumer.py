import numpy as np
import json
from confluent_kafka import Consumer, KafkaError
import logging
import requests
import time
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import os
import sys
import socket

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Determine mode from environment variable
mode = os.getenv('MODE', 'local').lower()
env_page = os.getenv('ENV_PAGE', 'default')

# Function to read remote server URL from configuration file
def get_remote_server_url():
    try:
        with open('host_dith.conf', 'r') as f:
            line = f.readline().strip()
            if not line:
                return None
            ip, port = line.split(':')
            return f"http://{ip}:{port}/update_frame"
    except Exception as e:
        logger.error(f"Error reading host_dith.conf: {e}")
        return None

if mode == 'remote':
    server_url = get_remote_server_url()
    if not server_url:
        logger.error("Cannot determine remote server URL. Exiting.")
        sys.exit(1)
elif mode == 'local':
    # Use the name of the service from docker-compose.yml
    server_url = "http://flask_server:5000/update_frame"
else:
    logger.error(f"Unknown mode: {mode}. Exiting.")
    sys.exit(1)

# Log the server URL for debugging
logger.info(f"Sending frames to {server_url}")

def create_black_image_with_text(text):
    """Create a black image with the given text."""
    img = Image.new('RGB', (1280, 720), color=(0, 0, 0))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/workspace/server_consumer/static/fonts/amazdoomleft.ttf", 100)
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

def send_frame_to_server(image_base64, epoch, loss, mode, mean_reward):
    """Send frame and metadata to the server via HTTP."""
    try:
        response = requests.post(
            server_url,
            json={
                'image': image_base64,
                'epoch': epoch,
                'loss': loss,
                'mode': mode,
                'meanReward': mean_reward,
                'env_page': env_page  # Add env_page to the payload
            }
        )
        if response.status_code == 200:
            logger.info("Frame sent successfully.")
        else:
            logger.error(f"Failed to send frame: {response.status_code} {response.text}")
    except Exception as e:
        logger.error(f"Error sending frame: {e}")

def consume_kafka_messages():
    """Consume messages from the 'doom_screen' Kafka topic."""
    consumer_config = {
        'bootstrap.servers': 'kafka:9092',  # Correct address
        'group.id': 'flask-consumer-group',
        'auto.offset.reset': 'earliest'
    }
    consumer = Consumer(consumer_config)
    consumer.subscribe(['doom_screen'])
    logger.info("Kafka consumer subscribed to 'doom_screen'")

    last_message_time = time.time()

    while True:
        try:
            msg = consumer.poll(timeout=0.5)
            current_time = time.time()
            if current_time - last_message_time > 2:
                logger.info("No messages received for 2 seconds. Sending default black image.")
                image_base64 = create_black_image_with_text("DITH isn't learning right now")
                send_frame_to_server(image_base64, 'Undefined', 'NaN', 'Undefined', 'NaN')
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
                image_base64 = message_data.get('image')
                epoch = message_data.get('epoch', 'Undefined')
                loss = message_data.get('loss', 'NaN')
                mode = message_data.get('mode', 'Unknown')
                mean_reward = message_data.get('meanReward', 'NaN')
                if not image_base64:
                    logger.info("Received empty image. Sending default black image.")
                    image_base64 = create_black_image_with_text("DITH isn't learning right now")
                else:
                    try:
                        image_data = base64.b64decode(image_base64)
                        img = Image.open(io.BytesIO(image_data))
                        img.verify()
                    except Exception as img_error:
                        logger.warning("Received an invalid image. Sending default black image.")
                        image_base64 = create_black_image_with_text("DITH isn't learning right now")
                send_frame_to_server(image_base64, epoch, loss, mode, mean_reward)
                last_message_time = time.time()
                logger.info("Frame and metadata sent successfully.")
            except Exception as decode_error:
                logger.error(f"Error decoding message: {decode_error}")
        except Exception as e:
            logger.error(f"Error in consume_kafka_messages: {e}")
            logger.info("Retrying connection to Kafka in 5 seconds...")
            time.sleep(5)
    consumer.close()
    logger.info("Kafka consumer closed.")

def send_initial_image():
    """Send a black image initially before subscribing to Kafka."""
    logger.info("Sending initial black image...")
    image_base64 = create_black_image_with_text("DITH isn't learning right now")
    send_frame_to_server(image_base64, 'Undefined', 'NaN', 'Undefined', 'NaN')
    logger.info("Initial black image sent successfully.")

if __name__ == '__main__':
    # Step 1: Send initial black image
    send_initial_image()

    # Step 2: Start consuming Kafka messages
    consume_kafka_messages()
