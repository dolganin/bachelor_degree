from flask import Flask, render_template
from flask_socketio import SocketIO
import base64
import cv2
import numpy as np
from confluent_kafka import Consumer, KafkaError
import logging
import queue
import threading
import time

# Настройка логирования
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Инициализация Flask-приложения и Flask-SocketIO
app = Flask(__name__)
socketio = SocketIO(app)

# Очередь для хранения кадров
frame_queue = queue.Queue(maxsize=1000)  # Ограничение размера очереди для предотвращения переполнения

def update_frame(frame):
    """
    Обновление кадра для отображения. Эта функция вызывается для передачи новых кадров.
    """
    try:
        # Убедимся, что frame — это изображение в правильном формате (например, 3 канала BGR)
        if frame.shape[2] == 3:  # Проверяем, что изображение цветное
            ret, buffer = cv2.imencode('.jpg', frame)  # Кодируем в формат JPEG
            if ret:
                # Конвертируем изображение в base64 для передачи через сокет
                encoded_image = base64.b64encode(buffer).decode('utf-8')
                try:
                    # Проверяем, заполнена ли очередь
                    if frame_queue.full():
                        # Удаляем старый кадр из начала очереди
                        dropped_frame = frame_queue.get()
                        logger.info("Dropped a frame from the queue to make space.")
                    frame_queue.put(encoded_image, timeout=1)  # Добавляем кадр в очередь
                    logger.info("Frame updated and encoded successfully.")
                except queue.Full:
                    logger.warning("Frame queue is still full. Dropping frame.")
            else:
                logger.warning("Failed to encode frame to JPEG.")
    except Exception as e:
        logger.error(f"Error in update_frame: {e}")


def send_frames():
    """
    Функция для отправки кадров клиентам через сокет.
    """
    try:
        while True:
            try:
                frame = frame_queue.get(timeout=0.03)  # Получаем кадр из очереди
                if frame:
                    socketio.emit('new_frame', {'image': frame})
                    logger.debug("Frame sent to client.")
            except queue.Empty:
                pass  # Нет новых кадров, продолжаем ожидание
            socketio.sleep(0.03)  # Отправляем кадры с задержкой в 30ms (~33 FPS)
    except Exception as e:
        logger.error(f"Error in send_frames: {e}")

def consume_kafka_messages():
    """
    Функция для потребления сообщений из Kafka топика 'doom_screen'.
    """
    consumer_config = {
        'bootstrap.servers': '192.168.3.2:9092',  # Замените на правильный IP-адрес брокера
        'group.id': 'flask-consumer-group',
        'auto.offset.reset': 'earliest'
    }

    while True:
        try:
            consumer = Consumer(consumer_config)
            consumer.subscribe(['doom_screen'])
            logger.info("Kafka consumer subscribed to 'doom_screen'")

            while True:
                msg = consumer.poll(timeout=1.0)
                if msg is None:
                    continue

                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        # Достигли конца партиции, продолжаем
                        continue
                    else:
                        logger.error(f"Kafka Error: {msg.error()}")
                        break

                # Получаем данные и декодируем изображение
                image_data = msg.value()
                logger.debug(f"Received message of size: {len(image_data)} bytes.")
                try:
                    frame = np.frombuffer(image_data, np.uint8).reshape(480, 640, 3)
                    if frame is not None:
                        update_frame(frame)
                        logger.info("Frame decoded and updated successfully.")
                    else:
                        logger.warning("Failed to decode frame.")
                except Exception as decode_error:
                    logger.error(f"Error decoding frame: {decode_error}")

        except Exception as e:
            logger.error(f"Error in consume_kafka_messages: {e}")
            logger.info("Retrying connection to Kafka in 5 seconds...")
            time.sleep(5)  # Ждём перед повторной попыткой подключения
        finally:
            try:
                consumer.close()
                logger.info("Kafka consumer closed.")
            except:
                pass
            time.sleep(5)  # Ждём перед повторной попыткой подключения

@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    logger.info("Client disconnected")

@app.route('/')
def index():
    """
    Маршрут для рендеринга основной страницы с отображением видео.
    """
    return render_template('index_socket.html')  # Убедитесь, что этот шаблон существует внутри папки 'templates'

if __name__ == '__main__':
    # Запуск фоновых задач
    logger.info("Starting background tasks...")
    socketio.start_background_task(send_frames)
    logger.info("Started send_frames task.")
    socketio.start_background_task(consume_kafka_messages)
    kafka_thread = threading.Thread(target=consume_kafka_messages)
    kafka_thread.start()
    logger.info("Started consume_kafka_messages task.")

    # Запуск приложения Flask с WebSocket'ами   
    logger.info("Running Flask server...")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)  # Используем 0.0.0.0 для доступности извне
