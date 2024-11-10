import os
import random
import torch
import pickle
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity: int = 10000, momentum: float = 0.995):
        """
        Инициализация буфера воспроизведения с затуханием.

        Args:
            capacity (int): Максимальная емкость буфера.
            momentum (float): Коэффициент затухания для старых значений, по умолчанию 0.995.
        """
        self.capacity = capacity  # Сохраняем емкость для повторной инициализации
        self.momentum = momentum
        self.buffer = deque(maxlen=capacity)
        self.dump_path = "dump_buffer.bin"

    def push(self, state: np.ndarray, action: int, next_state: np.ndarray, reward: float,
             combined_reward: float, action_log_prob: np.ndarray, done: bool):
        """
        Добавление нового перехода в буфер с применением коэффициента затухания для старых элементов.

        Args:
            state (np.ndarray): Текущее состояние.
            action (int): Действие агента.
            next_state (np.ndarray): Следующее состояние.
            reward (float): Награда за текущее действие.
            combined_reward (float): Комбинированная награда.
            action_log_prob (np.ndarray): Логарифм вероятности действия.
            done (bool): Флаг завершения эпизода.
        """
        # Применение затухания для всех предыдущих переходов
        self.buffer = deque([(s * self.momentum, a * self.momentum, ns * self.momentum, r * self.momentum,
                              cr * self.momentum, alp * self.momentum, d) for s, a, ns, r, cr, alp, d in self.buffer],
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
            torch.stack([torch.FloatTensor(state) for state in states]),
            torch.stack([torch.FloatTensor(action) for action in actions]),
            torch.stack([torch.FloatTensor(next_state) for next_state in next_states]),
            torch.FloatTensor(rewards),
            torch.FloatTensor(combined_rewards),
            torch.FloatTensor(action_log_probs),
            torch.BoolTensor(dones)
        )

    def dump(self):
        """
        Сохраняет содержимое буфера на диск в файл dump_buffer.bin и полностью удаляет буфер из памяти.
        """
        # Убедимся, что файл существует
        with open(self.dump_path, 'ab') as file:
            pass  # Просто создаем файл, если его нет
        with open(self.dump_path, 'wb') as file:
            pickle.dump(self.buffer, file)
        print("Buffer has been temporarily saved to disk as 'dump_buffer.bin'.")

        # Полное удаление буфера и его элементов из памяти
        for entry in list(self.buffer):
            for item in entry:
                del item  # Удаление каждого элемента перехода
            del entry  # Удаление самого перехода
        self.buffer.clear()  # Очистка буфера
        del self.buffer  # Удаление ссылки на буфер
        print("Buffer and its contents have been deleted from memory to free up RAM.")

    def load(self):
        """
        Загружает содержимое буфера из файла dump_buffer.bin в ОЗУ.
        После загрузки файл удаляется.
        """
        if os.path.exists(self.dump_path):
            with open(self.dump_path, 'rb') as file:
                self.buffer = pickle.load(file)
            os.remove(self.dump_path)
            print("Buffer has been loaded from disk and removed from 'dump_buffer.bin'.")
        else:
            # Повторная инициализация буфера, если файл отсутствует
            self.buffer = deque(maxlen=self.capacity)
            print("No buffer file found to load. Initialized a new empty buffer.")

    def __len__(self):
        return len(self.buffer) if self.buffer is not None else 0
