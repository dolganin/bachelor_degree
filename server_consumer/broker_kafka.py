import json
import numpy as np
from confluent_kafka import Producer
import base64
import cv2
import logging

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Kafka producer configuration
producer_config = {
    'bootstrap.servers': 'localhost:9092',  # Update with the correct broker address
    'message.max.bytes': 1000000,  # Increase if larger messages are expected
    'compression.type': 'snappy'  # Optional: enable compression
}

producer = Producer(producer_config)

def delivery_report(err, msg):
    """Callback for message delivery."""
    if err:
        logger.error(f"Message delivery failed: {err}")
    else:
        logger.info(f"Message delivered to {msg.topic()} [{msg.partition()}] at offset {msg.offset()}")

def publish_data(array: np.ndarray, epoch: int, loss: float, mode: str, mean_reward: float):
    """Publish data (numpy array and metadata) to Kafka."""
    try:
        # Ensure the array is in the correct format
        if not isinstance(array, np.ndarray) or array.dtype != np.uint8:
            logger.error("Invalid image array. Expected a numpy uint8 array.")
            return

        # Encode the numpy array to JPG and then to base64
        _, buffer = cv2.imencode('.jpg', array)
        array_base64 = base64.b64encode(buffer).decode('utf-8')

        # Ensure data types are correct
        epoch = int(epoch)
        loss = float(loss)
        mean_reward = float(mean_reward)
        if not isinstance(mode, str):
            logger.error("Mode must be a string.")
            return

        # Construct the message data
        message_data = {
            'image': array_base64,
            'epoch': epoch,
            'loss': loss,
            'mode': mode,
            'meanReward': mean_reward
        }

        # Convert to JSON string
        message_json = json.dumps(message_data)

        # Publish the message to the 'doom_screen' topic
        producer.produce('doom_screen', value=message_json, callback=delivery_report)

    except Exception as e:
        logger.error(f"Error publishing message: {e}")

# Optional: Periodically flush the producer
# def flush_producer():
#     producer.flush()
#     # Schedule the next flush
#     # threading.Timer(60, flush_producer).start()

# Call flush_producer() if you want periodic flushes