document.addEventListener("DOMContentLoaded", function() {
    var socket = io();  // Подключение к WebSocket
    
    socket.on('new_frame', function(data) {
        const host = (data && data.host) ? data.host.toString().trim().toLowerCase() : 'unknown';
        console.log('Received new frame from host:', host);
        
        if (1 === 1) {
            var img = document.getElementById('stream');
            img.src = 'data:image/png;base64,' + data.image;
            console.log('Updated image src:', img.src);
            
            document.getElementById('epoch').innerText = 'Epoch: ' + data.epoch;
            document.getElementById('loss').innerText = 'Loss: ' + data.loss;
            document.getElementById('meanReward').innerText = 'Mean Reward: ' + data.meanReward;
            document.getElementById('mode').innerText = 'Mode: ' + data.mode;
        } else {
            console.log('Ignoring frame from host:', host);
            if (!data.image) {
                console.log('Frame does not contain image data.');
            }
        }
    });
    
    socket.on('connect', function() {
        console.log('Connected to server');
        socket.emit('join', { page: 'tallas2' });
        console.log('Joined tallas2 room');
    });
    
    socket.on('disconnect', function() {
        console.log('Disconnected from server');
    });
    
    socket.on('connect_error', function(error) {
        console.error('Connection Error:', error);
    });
});