# app.py

from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO
import logging

# Настройка логирования
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def index():
    """Маршрут для рендеринга основной страницы с отображением видео."""
    return render_template('advanced_socket.html')

@app.route('/update_frame', methods=['POST'])
def update_frame():
    """Эндпоинт для обновления кадра."""
    data = request.json
    if 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    image = data['image']
    # Отправка изображения через WebSocket
    socketio.emit('new_frame', {'image': image, 'loss': 'NaN', 'epoch': "Undefined", 'meanReward': 'NaN'})
    logger.debug("Frame received and sent to client.")
    
    return jsonify({'status': 'success'}), 200

@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    logger.info("Client disconnected")

if __name__ == '__main__':
    logger.info("Running Flask server...")
    socketio.run(app, host='0.0.0.0', port=5000)
