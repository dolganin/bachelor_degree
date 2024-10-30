from collections import deque
import random
import torch
import numpy as np

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
    
    def push(self, state: np.ndarray, action: int, next_state: np.ndarray, reward: float, \
             combined_reward: float, action_log_prob: np.ndarray, done: bool):
        """
            Добавление нового перехода в буфер с применением коэффициента затухания для старых элементов.

            Args:
                state (np.ndarray): Текущее состояние.
                action (np.ndarray): Действие агента.
                next_state (np.ndarray): Следующее состояние.
                reward (float): Награда за текущее действие.
                combined_reward (float): Комбинированная награда.
                action_log_prob (np.ndarray): Логарифм вероятности действия.
                done (bool): Флаг завершения эпизода.
            """
        # Применение затухания для всех предыдущих переходов
        self.buffer = deque([(s * self.momentum, a * self.momentum, ns * self.momentum, r * self.momentum, \
                              cr*self.momentum, alp * self.momentum, d) for s, a, ns, r, cr, alp, d  in self.buffer], 
                            maxlen=self.buffer.maxlen)
        
        # Добавление нового перехода
        self.buffer.append((state, action, next_state, reward, combined_reward, action_log_prob, done))
    
    def sample(self, batch_size: int):
        """
        Сэмплирование батча из буфера.

        Args:
            batch_size (int): Размер батча.

        Returns:
            Tuple[torch.Tensor, ...]: Батч данных.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, next_states, rewards, combined_rewards, action_log_probs, dones = zip(*batch)

        return (
            torch.stack([torch.FloatTensor(state) for state in states]),            # Преобразуем каждый элемент
            torch.stack([torch.FloatTensor(action) for action in actions]),                                # Оборачиваем action в тензор (batch_size, 1)
            torch.stack([torch.FloatTensor(next_state) for next_state in next_states]),
            torch.FloatTensor(rewards),
            torch.FloatTensor(combined_rewards),
            torch.FloatTensor(action_log_probs),
            torch.BoolTensor(dones)
        )



    
    def __len__(self):
        return len(self.buffer)
