document.addEventListener("DOMContentLoaded", function() {
    // Specify the WebSocket URL if different from the current domain
    var socket = io('http://aurora.example.com'); 

    socket.on('new_frame', function(data) {
        // Ensure data format matches expectations
        var img = document.getElementById('stream');
        img.src = 'data:image/jpeg;base64,' + data.image;

        document.getElementById('epoch').innerText = 'Epoch: ' + data.epoch;
        document.getElementById('loss').innerText = 'Loss: ' + data.loss;
        document.getElementById('meanReward').innerText = 'Mean Reward: ' + data.meanReward;
        document.getElementById('mode').innerText = 'Mode: ' + data.mode;
    });

    socket.on('connect', function() {
        console.log('Connected to server');
        // Join the correct room for Aurora
        socket.emit('join', { page: 'aurora_room' });
        console.log('Joined aurora_room');
    });

    socket.on('disconnect', function() {
        console.log('Disconnected from server');
    });

    socket.on('connect_error', function(error) {
        console.error('Connection Error:', error);
    });
});