from abc import ABC, abstractmethod
import torch
import numpy as np

class RLAgent(ABC):
    @abstractmethod
    def __init__(self):
        """
        Инициализация базового агента.

        Args:
            action_size (int): Количество возможных действий.
            memory_size (int): Размер буфера памяти.
            batch_size (int): Размер батча для обучения.
            discount_factor (float): Коэффициент дисконтирования.
            lr (float): Скорость обучения.
            device (torch.device): Устройство для вычислений.
            model_savefile (str): Путь для сохранения модели.
        """
        pass

    @abstractmethod
    def get_action(self, state: np.ndarray):
        """
        Выбор действия на основе текущего состояния.

        Args:
            state (np.ndarray): Текущее состояние среды.

        Returns:
            tuple: Выбранное действие и логарифм вероятности действия.
        """
        pass

    @abstractmethod
    def append_memory(self, state: np.ndarray, action: int, reward: float, 
                      next_state: np.ndarray, done: bool):
        """
        Добавление перехода в буфер памяти.

        Args:
            state (np.ndarray): Текущее состояние.
            action (int): Действие агента.
            reward (float): Полученная награда.
            next_state (np.ndarray): Следующее состояние после действия.
            done (bool): Флаг завершения эпизода.
        """
        pass

    @abstractmethod
    def train_agent(self):
        """
        Обучение агента на основе собранных данных из буфера памяти.

        Returns:
            float: Значение функции потерь.
        """
        pass

    @abstractmethod
    def save_model(self, path: str):
        """
        Сохранение модели агента на диск.

        Args:
            path (str): Путь для сохранения модели.
        """
        pass

    @abstractmethod
    def load_model(self, path: str):
        """
        Загрузка сохранённой модели агента с диска.

        Args:
            path (str): Путь для загрузки модели.
        """
        pass

    @abstractmethod
    def update_target_net(self):
        """
        Обновление целевой сети, если это необходимо.
        """
        pass
