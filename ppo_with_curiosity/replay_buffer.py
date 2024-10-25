from collections import deque
import random
import torch

class ReplayBuffer:
    def __init__(self, capacity: int = 10000, momentum: float = 0.995):
        """
        Инициализация буфера воспроизведения с затуханием.

        Args:
            capacity (int): Максимальная емкость буфера.
            momentum (float): Коэффициент затухания для старых значений, по умолчанию 0.995.
        """
        self.buffer = deque(maxlen=capacity)
        self.momentum = momentum
    
    def push(self, state, action, next_state):
        """
        Добавление нового перехода в буфер с применением коэффициента затухания для старых элементов.

        Args:
            state (np.ndarray): Текущее состояние.
            action (np.ndarray): Действие агента.
            next_state (np.ndarray): Следующее состояние.
        """
        # Применение затухания для всех предыдущих переходов
        self.buffer = deque([(s * self.momentum, a * self.momentum, ns * self.momentum) for s, a, ns in self.buffer], 
                            maxlen=self.buffer.maxlen)
        
        # Добавление нового перехода
        self.buffer.append((state, action, next_state))
    
    def sample(self, batch_size: int):
        """
        Сэмплирование батча из буфера.

        Args:
            batch_size (int): Размер батча.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Батч состояний, действий и следующих состояний.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, next_states = zip(*batch)
        return torch.FloatTensor(states), torch.FloatTensor(actions), torch.FloatTensor(next_states)
    
    def __len__(self):
        return len(self.buffer)
