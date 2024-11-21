document.addEventListener("DOMContentLoaded", function() {
    var socket = io();  // Подключение к WebSocket
    
    socket.on('new_frame', function(data) {
        console.log('Received new frame:', data);
        var img = document.getElementById('stream');
        img.src = 'data:image/png;base64,' + data.image;
        console.log('Updated image src:', img.src);

        document.getElementById('epoch').innerText = 'Epoch: ' + data.epoch;
        document.getElementById('loss').innerText = 'Loss: ' + data.loss;
        document.getElementById('meanReward').innerText = 'Mean Reward: ' + data.meanReward;
        document.getElementById('mode').innerText = 'Mode: ' + data.mode;
    });

    socket.on('connect', function() {
        console.log('Connected to server');
        socket.emit('join', { page: 'doom_screen' });
        console.log('Joined doom_screen room');
    });

    socket.on('disconnect', function() {
        console.log('Disconnected from server');
    });

    socket.on('connect_error', function(error) {
        console.error('Connection Error:', error);
    });
});