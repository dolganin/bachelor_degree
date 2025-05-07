import { useState, useEffect } from 'react'
import { io } from 'socket.io-client'

// подключение к flask_server по имени контейнера
const socket = io('http://flask_server:5000');

export default function App() {
  const [hosts, setHosts] = useState([])
  const [room, setRoom] = useState(null)
  const [frame, setFrame] = useState({})

  useEffect(() => {
    socket.on('new_frame', data => {
      const host = data.env_page
      if (!hosts.includes(host)) setHosts(h => [...h, host])
      if (host === room) setFrame(data)
    })
  }, [hosts, room])

  const join = (h) => {
    setRoom(h)
    socket.emit('join', { page: h })
  }

  return (
    <div className="app-container">
      {!room && hosts.map(h =>
        <button key={h} onClick={() => join(h)}>{h}</button>
      )}
      {room && (
        <div>
          <button onClick={() => setRoom(null)}>← назад</button>
          <img src={'data:image/jpeg;base64,' + frame.image} alt="stream" />
          <div>Epoch: {frame.epoch}</div>
          <div>Reward: {frame.meanReward}</div>
        </div>
      )}
    </div>
  )
}
