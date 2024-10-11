from abc import ABC, abstractmethod
from time import time
from tqdm import trange
from preprocessing import preprocess
from typing import List
import numpy as np
from server_consumer.broker_kafka import publish_data
from torch.nn import Module
from video_logger import VideoLogger


class TrainerRL(ABC):
    def __init__(self, env, agent: Module, video_logger: VideoLogger=None, tensor_logger=None,  
                 device: str = "cpu", resolution: tuple = (30, 45), frame_repeat: int = 45, 
                 steps_per_epoch: int = 1000, actions: list = None, test_episodes_per_epoch: int = 1000) -> None:
        """
        Инициализация тренера.

        Args:
            env: Среда для обучения агента.
            agent: Объект агента, реализующий логику действий и обновления.
            config: Словарь или объект с конфигурациями для тренера.
        """
        self.env = env  # Среда, с которой агент взаимодействует
        self.agent = agent  # Агент, выполняющий действия и обучающийся
        self.current_step = 0  # Шаг обучения
        self.total_rewards = []  # Для хранения суммарных наград по эпизодам
        self.video_logger = video_logger
        self.tensor_logger = tensor_logger
        self.device = device
        self.resolution = resolution
        self.frame_repeat = frame_repeat
        self.steps_per_epoch = steps_per_epoch
        self.actions = actions
        self.test_episodes_per_epoch = test_episodes_per_epoch

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
        Оценка агента без обновления весов.

        Args:
            num_episodes: Количество эпизодов для оценки.
        """
        test_scores = []
        for _ in trange(self.test_episodes_per_epoch, leave=False):
            self.env.new_episode()
            while not self.env.is_episode_finished():
                state = preprocess(self.env.get_state().screen_buffer, resolution=self.resolution)

                temporal_state = np.array(self.env.get_state().screen_buffer, dtype=np.uint8)
                new_state = np.repeat(temporal_state[:, :, np.newaxis], 3, axis=2)


                best_action_index = self.agent.get_action(state)

                self.env.make_action(self.actions[best_action_index], self.frame_repeat)

                publish_data(array=new_state, epoch="Undefined", loss=float("NaN"), mean_reward=np.array(test_scores).mean(), mode="Test")
            r = self.env.get_total_reward()
            test_scores.append(r)

        test_scores = np.array(test_scores)
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

    def log_metrics(self, epoch: int = 0, mean_reward: float = float("NaN"), min_reward: float = float("NaN"), \
                    max_reward: float = float("NaN"), std_reward: float = float("NaN"), mean_loss: float = None) -> None:
        """
        Логгирование метрик обучения, таких как награды и потери.

        Args:
            episode: Текущий номер эпизода.
            reward: Суммарная награда за эпизод.
            loss: Потери модели (если есть).
        """

        self.tensor_logger.add_scalar('Test score minimum', min_reward, epoch)
        self.tensor_logger.add_scalar('Test score maximum', max_reward, epoch)
        self.tensor_logger.add_scalar('Test score mean', mean_reward, epoch)
        self.tensor_logger.add_scalar('Test score std', std_reward, epoch)
        self.tensor_logger.add_scalar('Mean Loss', mean_loss, epoch)

        print(f"Episode {epoch}: MeanReward = {mean_reward}, StdReward = {std_reward}, MeanLoss = {mean_loss}")

    def run(self, epochs: int = 0, evaluate_every: int = 1) -> None:
        """
        Полный процесс обучения с периодической оценкой.

        Args:
            num_episodes: Количество эпизодов для обучения.
            evaluate_every: Частота оценок после определенного количества эпизодов.
        """
        for epoch in range(epochs):
            start_time = time()
            test_scores = []
            print(f"\nEpoch #{epoch + 1}")

            # Запуск тренировки на одном эпизоде
            reward, loss_lst = self.train(epoch)
            self.total_rewards.append(reward)
            
            # Периодическая оценка
            if epoch % evaluate_every == 0:
                print("\nTesting...")
                test_scores = self.evaluate()
            
            # Логгирование результатов

            self.log_metrics(epoch, 
                             mean_reward=test_scores.mean(), 
                             std_reward=test_scores.std(), 
                             min_reward=test_scores.min(), 
                             max_reward=test_scores.max(),
                             mean_loss=loss_lst.mean())
            print("Total elapsed time: %.2f minutes" % ((time() - start_time) / 60.0))
        
        self.env.close()

# test(self.env, writter, epoch, self.agent, test_episodes_per_epoch=test_episodes_per_epoch, frame_repeat=frame_repeat, resolution=resolution, actions = actions)