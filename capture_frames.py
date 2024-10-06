from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import base64
import threading
import cv2
import numpy as np  # Импортируем NumPy для работы с массивами
import eventlet
from confluent_kafka import Consumer, KafkaError

# Для совместимости с Flask-SocketIO
eventlet.monkey_patch()

app = Flask(__name__)
socketio = SocketIO(app)

# Переменная для хранения текущего кадра и замок для многопоточности
current_frame = None
frame_lock = threading.Lock()

def update_frame(frame):
    """
    Обновление кадра для отображения. Эта функция вызывается для передачи новых кадров.
    """
    global current_frame
    if frame is not None:
        # Убедимся, что frame — это изображение в правильном формате (например, 3 канала BGR)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Конвертируем в BGR для OpenCV
        ret, buffer = cv2.imencode('.jpg', frame)  # Кодируем в формат JPEG
        if ret:
            with frame_lock:
                # Конвертируем изображение в base64 для передачи через сокет
                current_frame = base64.b64encode(buffer).decode('utf-8')

def send_frames():
    """
    Функция для отправки кадров клиентам через сокет.
    """
    while True:
        with frame_lock:
            if current_frame:
                socketio.emit('new_frame', {'image': current_frame})
        eventlet.sleep(0.03)  # Отправляем кадры с задержкой в 30ms

def consume_kafka_messages():
    """
    Функция для потребления сообщений из Kafka топика 'doom_state'.
    """
    consumer_config = {
        'bootstrap.servers': 'localhost:9092',  # Адрес Kafka-брокера
        'group.id': 'flask-consumer-group',
        'auto.offset.reset': 'earliest'
    }

    consumer = Consumer(consumer_config)
    consumer.subscribe(['doom_screen'])  # Подписываемся на топик 'doom_state'

    while True:
        msg = consumer.poll(timeout=1.0)  # Ожидаем новое сообщение из Kafka
        if msg is None:
            continue

        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                # Достигли конца партиции
                continue
            else:
                print(f"Kafka Error: {msg.error()}")
                break

        # Получаем данные и передаём их для обновления кадра
        frame_data = np.frombuffer(msg.value(), np.uint8)  # Преобразуем байты в NumPy массив
        frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)  # Декодируем массив в изображение

        # Передаем кадр в функцию для обновления
        update_frame(frame)

    consumer.close()

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@app.route('/')
def index():
    """
    Маршрут для рендеринга основной страницы с отображением видео.
    """
    return render_template('index_socket.html')

if __name__ == '__main__':
    # Поток для отправки кадров через сокет
    send_thread = threading.Thread(target=send_frames)
    send_thread.daemon = True
    send_thread.start()

    # Поток для потребления сообщений из Kafka
    kafka_thread = threading.Thread(target=consume_kafka_messages)
    kafka_thread.daemon = True
    kafka_thread.start()

    # Запуск приложения Flask с WebSocket'ами
    socketio.run(app, host='192.168.3.2', port=5000)
