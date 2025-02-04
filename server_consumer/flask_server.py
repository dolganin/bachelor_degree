from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, join_room
import logging
import os

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
socketio = SocketIO(app)

# Determine mode from environment variable
mode = os.environ.get('MODE', 'local').lower()
env_page = os.environ.get('ENV_PAGE', 'default_page').lower()  # Get the page from the environment variable

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
    return render_template('stream.html', page_name=env_page)

@app.route('/aurora')
def aurora():
    """Route for aurora stream."""
    return render_template('stream.html', page_name=env_page)

@app.route('/apollo2')
def apollo2():
    """Route for apollo2 stream."""
    return render_template('stream.html', page_name=env_page)

@app.route('/update_frame', methods=['POST'])
def update_frame():
    """Endpoint to update the frame."""
    data = request.json
    required_fields = ['image', 'epoch', 'mode', 'loss', 'meanReward', 'hostname']
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
    hostname = data['hostname']

    try:
        # Emit frame to room determined by ENV_PAGE
        socketio.emit('new_frame', 
                      {'image': image, 'loss': loss, 'epoch': epoch, 
                       'meanReward': meanReward, 'mode': mode}, 
                      room=env_page)
    except Exception as e:
        logger.error(f"Error processing frame data: {e}")
        socketio.emit('new_frame', 
                      {'image': image, 'loss': "NaN", 'epoch': "Undefined", 
                       'meanReward': "NaN", 'mode': mode}, 
                      room=env_page)

    logger.debug(f"Frame received and sent to {env_page} room.")
    return jsonify({'status': 'success'}), 200

@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')

@socketio.on('join')
def handle_join(data):
    """Handle room joining."""
    # Always join the room determined by ENV_PAGE
    join_room(env_page)
    logger.info(f"Client joined room {env_page}")

@socketio.on('disconnect')
def handle_disconnect():
    logger.info("Client disconnected")

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
