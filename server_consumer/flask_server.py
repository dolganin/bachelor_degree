from flask import Flask, request, jsonify
from flask_socketio import SocketIO, join_room, leave_room
from flask_cors import CORS
import logging
import os

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

mode = os.environ.get('MODE', 'local').lower()
if mode == 'remote':
    logger.info("Remote mode: server not started.")
    exit(0)
logger.info("Local mode: starting on port 5000.")

@app.route('/update_frame', methods=['POST'])
def update_frame():
    data = request.json or {}
    required = ['image', 'epoch', 'mode', 'meanReward', 'env_page']
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({'error': 'Missing fields: ' + ','.join(missing)}), 400

    try:
        mean_reward = round(data.get('meanReward', 0), 2)
    except:
        mean_reward = "NaN"

    payload = {
        'image': data['image'],
        'epoch': data['epoch'],
        'meanReward': mean_reward,
        'mode': data['mode'],
        'env_page': data['env_page']
    }

    try:
        socketio.emit('new_frame', payload, room=data['env_page'])
        logger.debug(f"Frame sent to room '{data['env_page']}'.")
    except Exception as e:
        logger.error(f"Emit error: {e}")

    return jsonify({'status': 'success'}), 200

@socketio.on('connect')
def on_connect():
    logger.info("Client connected")

@socketio.on('join')
def on_join(data):
    room = data.get('page')
    if room:
        join_room(room)
        logger.info(f"Client joined room '{room}'")
    else:
        logger.warning("Join called without 'page'")

@socketio.on('leave')
def on_leave(data):
    room = data.get('page')
    if room:
        leave_room(room)
        logger.info(f"Client left room '{room}'")
    else:
        logger.warning("Leave called without 'page'")

@socketio.on('disconnect')
def on_disconnect():
    logger.info("Client disconnected")

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
