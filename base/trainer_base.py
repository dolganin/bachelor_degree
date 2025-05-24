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
    def evaluate(self, log_video: bool = False, send_frames: bool = False) -> np.ndarray:
        test_scores = []
        obs = self.env.reset()
        n_envs = self.env.n_envs
        rewards = [0.0 for _ in range(n_envs)]
        active = [True] * n_envs  # отслеживаем, какие среды ещё не завершили эпизод

        while any(active):
            batch_states = [preprocess(o, resolution=self.resolution) for o in obs]
            actions, _ = zip(*[self.agent.get_action(s) for s in batch_states])

            if self.actions:
                selected = []
                for a in actions:
                    idx = int(torch.argmax(torch.tensor(a)).item())
                    selected.append(self.actions[idx])
            else:
                selected = actions

            obs, step_rewards, dones, infos = self.env.step(selected)

            for i in range(n_envs):
                if active[i]:
                    rewards[i] += step_rewards[i]

                    if log_video or send_frames:
                        frame = np.array(obs[i], dtype=np.uint8)
                        if frame.shape[-1] == 3:
                            frame = frame[..., ::-1]
                        frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)

                        if log_video:
                            self.video_logger.add_frame(frame)
                        if send_frames:
                            publish_data(array=frame, epoch="Validation", loss=float("NaN"),
                                        mean_reward=0.0, mode="Test")

                    if dones[i]:
                        test_scores.append(rewards[i])
                        active[i] = False

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
                    mean_loss: float = None, policy_loss: float = None, value_loss: float = None) -> None:
        """
        Логгирование метрик обучения, таких как награды и потери.

        Args:
            episode: Текущий номер эпизода.
            reward: Суммарная награда за эпизод.
            loss: Потери модели (если есть).
        """
        self.wandb_logger.log({
            'Mean Policy loss': policy_loss,
            'Mean Value loss': value_loss,
            'Test score mean': mean_reward,
            'Test score std': std_reward,
            'Mean loss': mean_loss,
            'Epoch': epoch
        })

        print("Metrics of model was logged to wandb!")

    def run(self, total_steps: int = 500000, validate_every_split: int = 5, batch_size: int = 64) -> None:
        """
        Запуск обучения с валидацией каждые (total_steps / validate_every_split) шагов.
        """
        steps_per_val = total_steps // validate_every_split
        steps_completed = 0
        epoch = 0

        while steps_completed < total_steps:
            print(f"[RUN] Эпоха {epoch+1} — старт обучения на {steps_per_val} шагов...")

            reward, loss_lst = self.train(total_steps=steps_per_val, batch_size=batch_size)
            self.total_rewards.append(reward)

            policy_loss = np.array(loss_lst["policy_loss"]).mean()
            value_loss = np.array(loss_lst["value_loss"]).mean()
            mean_loss = (policy_loss + value_loss) / 2

            print(f"[RUN] Эпоха {epoch+1} — обучение завершено, запускается валидация...")

            test_scores = self.evaluate()
            avg_reward = test_scores.mean()
            std_reward = test_scores.std()

            self.log_metrics(
                epoch=epoch,
                mean_reward=avg_reward,
                std_reward=std_reward,
                policy_loss=policy_loss,
                value_loss=value_loss,
                mean_loss=mean_loss
            )

            self.avaluator.evaluate_and_save(
                trainer=self,
                mean_reward=avg_reward,
                std_reward=std_reward
            )

            print(f"[RUN] Эпоха {epoch+1} завершена. Прогресс: {steps_completed + steps_per_val}/{total_steps} шагов.")
            steps_completed += steps_per_val
            epoch += 1

        self.env.close()
        print("[RUN] Обучение завершено. Среда закрыта.")

