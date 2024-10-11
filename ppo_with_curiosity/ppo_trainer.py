# PPO_Trainer.py

import torch
import numpy as np
from tqdm import trange

from base.trainer_base import TrainerRL  # Предполагается, что TrainerRL определён в base/trainer_base.py
from preprocessing import preprocess
from server_consumer.broker_kafka import publish_data

from ppo_with_curiosity.ppo_agent import PPOAgent  # Импорт PPOAgent из PPO_Agent.py


class PPOTrainer(TrainerRL):
    def __init__(self, env, agent: Module, video_logger: VideoLogger=None, tensor_logger=None,  
                 device: str = "cpu", resolution: tuple = (30, 45), frame_repeat: int = 45, 
                 steps_per_epoch: int = 1000, actions: list = None, test_episodes_per_epoch: int = 1000, model_savefile: str = None):
        """
        Инициализация PPOTrainer с настройками для PPO with Curiosity.

        Args:
            env_config (dict): Конфигурация среды.
            agent_config (dict): Конфигурация агента (сети и оптимизаторы).
            logger_config (dict): Конфигурация логгера.
            **kwargs: Дополнительные аргументы.
        """
        super(PPOTrainer, self).__init__()
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
        self.model_savefile = model_savefile if model_savefile is not None else "model.pth"

        
    def train(self, episode: int, steps_per_epoch: int = 1000):
        """
        Основной цикл обучения агента для одного эпизода с использованием PPO with Curiosity.

        Args:
            episode (int): Номер текущего эпизода.
            steps_per_epoch (int, optional): Количество шагов за эпизод. По умолчанию 1000.

        Returns:
            tuple: Суммарная награда за эпизод и массив значений потерь.
        """
        loss_lst = []
        self.env.new_episode()
        train_scores = []
        global_step = 0
        total_reward = 0.0
        total_intrinsic = 0.0
        
        for _ in trange(steps_per_epoch, leave=False, desc=f"Episode {episode+1}"):
            # Получение и предобработка текущего состояния
            raw_state = self.env.get_state().screen_buffer
            state = preprocess(raw_state, resolution=self.resolution)
            
            # Логирование видеофрейма
            temporal_state = np.array(raw_state, dtype=np.uint8)
            new_state = np.repeat(temporal_state[:, :, np.newaxis], 3, axis=2)
            self.video_logger.add_frame(new_state)  # Добавляем картинку в лог
            
            # Выбор действия
            action, action_log_prob = self.agent.get_action(state)
            if self.actions is None:
                selected_action = action
            else:
                selected_action = self.actions[action]
            
            # Выполнение действия в среде
            reward = self.env.make_action(selected_action, self.frame_repeat)
            done = self.env.is_episode_finished()
            total_reward += reward
            
            # Получение следующего состояния
            if not done:
                next_raw_state = self.env.get_state().screen_buffer
                next_state = preprocess(next_raw_state, resolution=self.resolution)
            else:
                next_state = np.zeros((1, self.resolution[0], self.resolution[1]), dtype=np.float32)
            
            # Вычисление внутреннего вознаграждения
            intrinsic_reward = self.agent.compute_intrinsic_reward(state, action, next_state)
            total_intrinsic += intrinsic_reward
            combined_reward = reward + self.agent.lambda_intrinsic * intrinsic_reward
            
            # Сохранение перехода в буфер
            self.agent.append_memory(state, action, reward, combined_reward, action_log_prob, next_state, done)
            
            # Добавление в Replay Buffer Forward Model
            self.agent.forward_replay_buffer.push(state, action, next_state)
            
            # Логирование данных (например, отправка в Kafka)
            publish_data(
                array=new_state, 
                epoch=episode, 
                loss=0.0,  # Изначально без потерь
                mean_reward=np.array(train_scores).mean() if train_scores else 0.0, 
                mode="Train"
            )
            
            global_step += 1
            
            # Обучение агента, если буфер заполнен
            if global_step > self.agent.batch_size and len(self.agent.memory) >= self.agent.batch_size:
                policy_loss, value_loss = self.agent.train()
                loss_lst.append((policy_loss, value_loss))
            
            # Завершение эпизода
            if done:
                total_episode_reward = self.env.get_total_reward()
                train_scores.append(total_episode_reward)
                self.env.new_episode()
                break  # Начать следующий эпизод
        
        # Обновление Forward Model
        forward_loss = self.agent.update_forward_model()
        if forward_loss > 0.0:
            loss_lst.append(('Forward Loss', forward_loss))
        
        # Логирование прогресса
        average_reward = np.array(train_scores).mean() if train_scores else 0.0
        average_policy_loss = np.mean([loss[0] for loss in loss_lst if isinstance(loss, tuple) and len(loss) == 2]) if loss_lst else 0.0
        average_value_loss = np.mean([loss[1] for loss in loss_lst if isinstance(loss, tuple) and len(loss) == 2]) if loss_lst else 0.0
        average_forward_loss = np.mean([loss[1] for loss in loss_lst if isinstance(loss, tuple) and len(loss) == 2]) if loss_lst else 0.0
        
        print(f"Episode {episode+1}, Reward: {total_reward:.2f}, Intrinsic: {total_intrinsic:.2f}, "
              f"Policy Loss: {average_policy_loss:.4f}, Value Loss: {average_value_loss:.4f}, "
              f"Forward Loss: {average_forward_loss:.4f}")
        
        return total_reward, np.array(loss_lst)  # Возвращаем суммарную награду и потери для логирования

    def save_model(self, path: str):
        """
        Сохранение моделей агента на диск.

        Args:
            path (str): Базовый путь для сохранения моделей (без расширений).
        """
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'value_net_state_dict': self.value_net.state_dict(),
            'forward_model_state_dict': self.forward_model.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'forward_optimizer_state_dict': self.forward_optimizer.state_dict(),
        }, f"{path}.pth")
        print(f"Models saved to {path}.pth")

    def load_model(self, path: str):
        """
        Загрузка сохранённых моделей агента с диска.

        Args:
            path (str): Базовый путь для загрузки моделей (без расширений).
        """
        checkpoint = torch.load(f"{path}.pth", map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.forward_model.load_state_dict(checkpoint['forward_model_state_dict'])
        
        self.policy_net.to(self.device)
        self.value_net.to(self.device)
        self.forward_model.to(self.device)
        
        # Загрузка состояний оптимизаторов
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        self.forward_optimizer.load_state_dict(checkpoint['forward_optimizer_state_dict'])
        
        print(f"Models and optimizers loaded from {path}.pth")

