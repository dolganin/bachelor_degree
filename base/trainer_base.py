from abc import ABC, abstractmethod
from time import time
import torch
from tqdm import trange
from utilities.preprocessing import preprocess
from typing import List
import numpy as np
from server_consumer.broker_kafka import publish_data
import cv2
from torch import argmax, Tensor
from colorama import Fore, Style


class TrainerRL(ABC):
    @abstractmethod
    def __init__(self) -> None:
        """
        Инициализация тренера.

        Args:
            env: Среда для обучения агента.
            agent: Объект агента, реализующий логику действий и обновления.
            config: Словарь или объект с конфигурациями для тренера.
        """
        pass

    @abstractmethod
    def train(self, epoch: int = 0, steps_per_epoch: int = 1000):
        """
        Основной цикл обучения агента в среде.

        Args:
            num_episodes: Количество эпизодов для обучения.
        """
        pass
    def evaluate(self) -> np.ndarray:
        """
        Оценка агента без обновления весов. Функция проходит по заданному количеству тестовых эпизодов,
        собирает видеофреймы и итоговые награды.
        
        Returns:
            np.ndarray: Массив итоговых наград по тестовым эпизодам.
        """
        test_scores = []
        for _ in trange(self.test_episodes_per_epoch, leave=False):
            self.env.new_episode()
            while not self.env.is_episode_finished():
                raw_state = self.env.get_state().screen_buffer
                state = preprocess(raw_state, resolution=self.resolution)
                
                # Логирование видеофрейма
                temporal_state = np.array(raw_state, dtype=np.uint8)
                if temporal_state.shape[-1] == 3:
                    temporal_state = temporal_state[..., ::-1]  # Если нужно поменять порядок каналов
                temporal_state = cv2.resize(temporal_state, (1280, 720), interpolation=cv2.INTER_LINEAR)
                self.video_logger.add_frame(temporal_state)
                
                # Выбор действия
                action, _ = self.agent.get_action(state)
                if self.actions is not None:
                    # Если имеется список действий, выбираем индекс максимального значения
                    action_tensor = torch.tensor(action)
                    selected_action_idx = int(torch.argmax(action_tensor).item())
                    selected_action = self.actions[selected_action_idx]
                else:
                    selected_action = action

                self.env.make_action(selected_action, self.frame_repeat)
                
                # Отправка фрейма (для логирования или визуализации)
                publish_data(
                    array=temporal_state,
                    epoch="Undefined",
                    loss=float("NaN"),
                    mean_reward=0.0,
                    mode="Test"
                )
            
            r = self.env.get_total_reward()
            test_scores.append(r)
        
        test_scores = np.array(test_scores)
        self.avaluator.evaluate_and_save(self, test_scores.mean(), test_scores.std())
        
        return test_scores


    @abstractmethod
    def save_model(self, filepath: str):
        """
        Сохранение текущей модели агента на диск.

        Args:
            filepath: Путь для сохранения модели.
        """
        pass

    @abstractmethod
    def load_model(self, filepath: str):
        """
        Загрузка модели агента с диска.

        Args:
            filepath: Путь для загрузки модели.
        """
        pass

    def log_metrics(self, epoch: int = 0, mean_reward: float = float("NaN"), std_reward: float = float("NaN"), \
                    mean_loss: float = None, forward_loss: float = None, \
                        policy_loss: float = None, value_loss: float = None) -> None:
        """
        Логгирование метрик обучения, таких как награды и потери.

        Args:
            episode: Текущий номер эпизода.
            reward: Суммарная награда за эпизод.
            loss: Потери модели (если есть).
        """
        self.wandb_logger.log({
            # 'Mean Forward loss': forward_loss,
            'Mean Policy loss': policy_loss,
            'Mean Value loss': value_loss,
            'Test score mean': mean_reward,
            'Test score std': std_reward,
            'Mean loss': mean_loss,
            'Epoch': epoch
        })

        print("Metrics of model was logged to tensorboard!")

    def run(self, epochs: int = 0, evaluate_every: int = 100) -> None:
        """
        Полный процесс обучения с периодической оценкой.
        """
        max_reward = 0.0

        with trange(epochs, desc="Training", unit="epoch", bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as pbar:
            for epoch in pbar:
                test_scores = []

                # Запуск тренировки на одном эпизоде
                reward, loss_lst = self.train(epoch, steps_per_epoch=self.steps_per_epoch)
                self.total_rewards.append(reward)

                # Обновление прогресс-бара с временем начала
                pbar.set_postfix(epoch=epoch + 1, reward=reward)

                # Периодическая оценка
                if epoch % evaluate_every == 0:
                    print(Fore.YELLOW + "\nTesting..." + Style.RESET_ALL)
                    test_scores = self.evaluate()
                    test_scores = np.array(test_scores)

                    # forward_loss = np.array(loss_lst["forward_loss"]).mean()
                    policy_loss = np.array(loss_lst["policy_loss"]).mean()
                    value_loss = np.array(loss_lst["value_loss"]).mean()
                    #mean_loss = (forward_loss + policy_loss + value_loss) / 3
                    mean_loss = (policy_loss + value_loss)/2

                    self.log_metrics(epoch, 
                                    mean_reward=test_scores.mean(), 
                                    std_reward=test_scores.std(),
                                    #forward_loss=forward_loss,
                                    policy_loss=policy_loss,
                                    value_loss=value_loss,
                                    mean_loss=mean_loss
                    )
                    
                pbar.update(1)

        self.env.close()