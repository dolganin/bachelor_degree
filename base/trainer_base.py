from abc import ABC, abstractmethod
from time import time
from tqdm import trange
from utilities.preprocessing import preprocess
from typing import List
import numpy as np
from server_consumer.broker_kafka import publish_data
import cv2
from torch import argmax, Tensor


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
        Оценка агента без обновления весов.

        Args:
            num_episodes: Количество эпизодов для оценки.
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
                    # Меняем порядок каналов с RGB на BGR, если необходимо
                    temporal_state = temporal_state[..., ::-1]  # Меняем порядок на BGR

                # Изменение размера изображения до 1280x720
                temporal_state = cv2.resize(temporal_state, (1280, 720), interpolation=cv2.INTER_LINEAR)
                
                #new_state = np.repeat(temporal_state[:, :, np.newaxis], 3, axis=2)
                self.video_logger.add_frame(temporal_state)
                
                action_distribution, _ = self.agent.get_action(state)
                action_distribution = Tensor(action_distribution) 
                selected_action_idx = int(argmax(action_distribution).item())

                self.env.make_action(self.actions[selected_action_idx], self.frame_repeat)
                

                publish_data(array=temporal_state, epoch="Undefined", loss=float("NaN"), mean_reward=np.array(test_scores).mean(), mode="Test")
            r = self.env.get_total_reward()
            test_scores.append(r)

        test_scores = np.array(test_scores)
        self.avaluator.evaluate_and_save(self, test_scores.mean(), test_scores.std(), self.agent.compute_total_loss())
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
        max_reward = 0.0
        for epoch in range(epochs):
            start_time = time()
            test_scores = []

            # Запуск тренировки на одном эпизоде
            reward, loss_lst = self.train(epoch, steps_per_epoch=self.steps_per_epoch)
            self.total_rewards.append(reward)
            
            # Периодическая оценка
            if epoch % evaluate_every == 0:
                print("\nTesting...")
                test_scores = self.evaluate()
            
            # Логгирование результатов
            test_scores = np.array(test_scores)

            self.log_metrics(epoch, 
                             mean_reward=test_scores.mean(), 
                             std_reward=test_scores.std(),
                             mean_loss=loss_lst.mean())
            print("Total elapsed time: %.2f minutes" % ((time() - start_time) / 60.0))
        
        self.env.close()