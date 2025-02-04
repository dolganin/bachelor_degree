from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, join_room, leave_room
import logging
import os

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
socketio = SocketIO(app)

# Determine mode from environment variable
mode = os.environ.get('MODE', 'local').lower()

if mode == 'remote':
    logger.info("Running in remote mode. Server not started.")
    # Exit the application
    exit(0)
elif mode == 'local':
    logger.info("Running in local mode. Server starting on port 5000.")
else:
    logger.warning(f"Unknown mode: {mode}. Defaulting to local mode.")

@app.route('/')
def index():
    """Route for the hub streams."""
    return render_template('hub/hub.html')

@app.route('/tallas2')
def tallas2():
    """Route for tallas2 stream."""
    return render_template('stream.html', server_name='Tallas2', page_name='tallas2')

@app.route('/aurora')
def aurora():
    """Route for aurora stream."""
    return render_template('stream.html', server_name='Aurora', page_name='aurora')

@app.route('/apollo2')
def apollo2():
    """Route for apollo2 stream."""
    return render_template('stream.html', server_name='Apollo2', page_name='apollo2')

@app.route('/update_frame', methods=['POST'])
def update_frame():
    """Endpoint to update the frame."""
    data = request.json
    required_fields = ['image', 'epoch', 'mode', 'loss', 'meanReward', 'env_page']
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return jsonify({'error': f'Missing fields: {", ".join(missing_fields)}'}), 400

    image = data['image']
    epoch = data['epoch']
    mode = data['mode']

    try:
        loss = round(data['loss'], 2)
        meanReward = round(data['meanReward'], 2)
    except:
        loss = "NaN"
        meanReward = "NaN"

    try:
        socketio.emit('new_frame',
                      {'image': image, 'loss': loss, 'epoch': epoch,
                       'meanReward': meanReward, 'mode': mode},
                      room=data['env_page'])
        logger.debug(f"Frame sent to {data['env_page']} room.")
    except Exception as e:
        logger.error(f"Error processing frame data: {e}")
        socketio.emit('new_frame',
                      {'image': image, 'loss': "NaN", 'epoch': "Undefined",
                       'meanReward': "NaN", 'mode': mode},
                      room=data['env_page'])

    return jsonify({'status': 'success'}), 200

@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')

@socketio.on('join')
def handle_join(data):
    page = data.get('env_page')
    if page:
        join_room(page)
        logger.info(f"Client joined room {page}")
    else:
        logger.warning("Client tried to join room without specifying page")

@socketio.on('leave')
def handle_leave(data):
    page = data.get('env_page')
    if page:
        leave_room(page)
        logger.info(f"Client left room {page}")
    else:
        logger.warning("Client tried to leave room without specifying page")

@socketio.on('disconnect')
def handle_disconnect():
    logger.info("Client disconnected")

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
