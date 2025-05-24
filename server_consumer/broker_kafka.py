import json
import numpy as np
from confluent_kafka import Producer
import base64
import cv2
import logging
import sys
import os
import contextlib

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Kafka producer configuration
producer_config = {
    'bootstrap.servers': 'localhost:9093',
    'message.max.bytes': 1000000,
    'compression.type': 'snappy',
    'log_level': 0  # Всё, что можем подавить напрямую
}

# Context manager to suppress stderr
@contextlib.contextmanager
def suppress_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

# Create producer with stderr suppressed
with suppress_stderr():
    producer = Producer(producer_config)

# Failure tracking
MAX_FAILURES = 30
failure_count = 0
kafka_enabled = True

def delivery_report(err, msg):
    """Callback for message delivery."""
    if err:
        logger.error(f"Message delivery failed: {err}")
    else:
        logger.info(f"Message delivered to {msg.topic()} [{msg.partition()}] at offset {msg.offset()}")


def publish_data(array: np.ndarray, epoch: int, loss: float, mode: str, mean_reward: float):
    """Publish data (numpy array and metadata) to Kafka."""
    global failure_count, kafka_enabled

    if not kafka_enabled:
        logger.warning("Kafka publishing disabled after too many failures.")
        return

    try:
        if not isinstance(array, np.ndarray) or array.dtype != np.uint8:
            logger.error("Invalid image array. Expected a numpy uint8 array.")
            return

        _, buffer = cv2.imencode('.jpg', array)
        array_base64 = base64.b64encode(buffer).decode('utf-8')

        try:
            epoch = int(epoch)
        except (ValueError, TypeError):
            epoch = 0
            logger.warning("Invalid epoch value. Setting to 0.")

        try:
            loss = float(loss)
        except (ValueError, TypeError):
            loss = 0.0
            logger.warning("Invalid loss value. Setting to 0.0.")

        try:
            mean_reward = float(mean_reward)
        except (ValueError, TypeError):
            mean_reward = 0.0
            logger.warning("Invalid mean reward value. Setting to 0.0.")

        if not isinstance(mode, str):
            logger.error("Mode must be a string.")
            return

        message_data = {
            'image': array_base64,
            'epoch': epoch,
            'loss': loss,
            'mode': mode,
            'meanReward': mean_reward
        }

        message_json = json.dumps(message_data)

        producer.produce('doom_screen', value=message_json, callback=delivery_report)
        producer.poll(0)
        failure_count = 0

    except Exception as e:
        failure_count += 1
        logger.error(f"Error publishing message: {e} (Failure {failure_count}/{MAX_FAILURES})")
        if failure_count >= MAX_FAILURES:
            kafka_enabled = False
            logger.critical("Kafka publishing disabled after exceeding maximum failure count.")
