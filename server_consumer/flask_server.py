from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO
import logging
import os

# Импортируем Kafka для отправки данных в удалённом режиме
from kafka import KafkaProducer
import json

# Настройка логирования
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Режим работы: локальный сервер или удалённый
MODE = os.getenv('MODE', 'local')  # 'local' или 'remote'

# Устанавливаем Kafka-продюсера для отправки данных в случае удалённого режима
if MODE == 'remote':
    kafka_producer = KafkaProducer(
        bootstrap_servers=['your_kafka_broker:9092'],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

# Инициализация Flask и SocketIO
app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def index():
    """Маршрут для хаба стримов."""
    return render_template('hub/hub.html')

@app.route('/tallas2')
def tallas2():
    """Маршрут для стрима tallas2."""
    return render_template('tallas2/tallas2.html')

@app.route('/aurora')
def aurora():
    """Маршрут для стрима aurora."""
    return render_template('aurora/aurora.html')

@app.route('/apollo2')
def apollo2():
    """Маршрут для стрима apollo2."""
    return render_template('apollo2/apollo2.html')

@app.route('/update_frame', methods=['POST'])
def update_frame():
    """Эндпоинт для обновления кадра."""
    data = request.json
    if 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    image = data['image']
    epoch = data['epoch']
    mode = data['mode']

    try:
        loss = round(data['loss'], 2)
        meanReward = round(data['meanReward'], 2)

        # Если сервер работает в локальном режиме
        if MODE == 'local':
            socketio.emit('new_frame', {'image': image, 'loss': loss, 'epoch': epoch, 'meanReward': meanReward, 'mode': mode})
        # Если сервер работает в удалённом режиме, отправляем через Kafka
        elif MODE == 'remote':
            frame_data = {'image': image, 'loss': loss, 'epoch': epoch, 'meanReward': meanReward, 'mode': mode}
            kafka_producer.send('frame_topic', frame_data)
            logger.debug("Frame sent to Kafka producer.")

    except Exception as e:
        logger.error(f"Error processing frame data: {e}")
        # В случае ошибки отправляем данные с NaN значениями
        error_data = {'image': image, 'loss': "NaN", 'epoch': "Undefined", 'meanReward': "NaN", 'mode': mode}
        
        if MODE == 'local':
            socketio.emit('new_frame', error_data)
        elif MODE == 'remote':
            kafka_producer.send('frame_topic', error_data)

    logger.debug("Frame received and processed.")
    
    return jsonify({'status': 'success'}), 200

@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    logger.info("Client disconnected")

if __name__ == '__main__':
    logger.info("Running Flask server...")

    if MODE == 'local':
        # Запускаем сервер Flask только если работает в локальном режиме
        socketio.run(app, host='0.0.0.0', port=5000)
    else:
        logger.info("Running in remote mode, no Flask server started.")
