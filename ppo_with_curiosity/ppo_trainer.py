import torch
import numpy as np
from tqdm import trange
from torch.nn import Module
from base.trainer_base import TrainerRL  # Предполагается, что TrainerRL определён в base/trainer_base.py
from utilities.preprocessing import preprocess
from server_consumer.broker_kafka import publish_data
from utilities.video_logger import VideoLogger
from ppo_with_curiosity.ppo_agent import PPOAgent  # Импорт PPOAgent из PPO_Agent.py
from .replay_buffer import ReplayBuffer  # Импортируем ReplayBuffer с затуханием
import cv2
from base.agent_evaluator import AgentEvaluator

class PPOTrainer(TrainerRL):
    def __init__(self, env, agent: Module, video_logger: VideoLogger=None, tensor_logger=None,  
                 device: str = "cpu", resolution: tuple = (30, 45), frame_repeat: int = 45, 
                 steps_per_epoch: int = 1000, actions: list = None, test_episodes_per_epoch: int = 1000, 
                 model_savefile: str = None, buffer_capacity: int = 10000, buffer_momentum: float = 0.995, 
                 agent_evaluator: AgentEvaluator = None):
        """
        Инициализация PPOTrainer с настройками для PPO with Curiosity.

        Args:
            env: Среда, с которой агент взаимодействует
            agent (Module): Агент, выполняющий действия и обучающийся
            video_logger (VideoLogger, optional): Логгер для видео.
            tensor_logger (optional): Логгер для тензорных данных.
            device (str): Устройство для вычислений, например, "cpu" или "cuda".
            resolution (tuple): Размер изображения, по умолчанию (30, 45).
            frame_repeat (int): Параметр повторения кадров, по умолчанию 45.
            steps_per_epoch (int): Количество шагов за эпизод, по умолчанию 1000.
            actions (list, optional): Список действий, доступных агенту.
            test_episodes_per_epoch (int): Количество тестовых эпизодов за эпоху.
            model_savefile (str, optional): Путь для сохранения модели.
            buffer_capacity (int): Размер буфера, по умолчанию 10000.
            buffer_momentum (float): Коэффициент затухания для старых значений, по умолчанию 0.995.
        """
        super(PPOTrainer, self).__init__()
        self.env = env
        self.agent = agent
        self.current_step = 0
        self.total_rewards = []
        self.video_logger = video_logger
        self.tensor_logger = tensor_logger
        self.device = device
        self.resolution = resolution
        self.frame_repeat = frame_repeat
        self.steps_per_epoch = steps_per_epoch
        self.actions = actions
        self.test_episodes_per_epoch = test_episodes_per_epoch
        self.model_savefile = model_savefile if model_savefile is not None else "model.pth"
        
        # Инициализация памяти с заданной емкостью и затуханием
        self.memory = ReplayBuffer(capacity=buffer_capacity, momentum=buffer_momentum)
        self.avaluator = agent_evaluator

    def train(self, episode: int, steps_per_epoch: int = 1000):
        """
        Основной цикл обучения агента для одного эпизода с использованием PPO with Curiosity.
        """
        loss_lst = []
        self.env.new_episode()
        train_scores = []
        global_step = 0
        total_reward = 0.0
        total_intrinsic = 0.0
        
        for _ in trange(steps_per_epoch, leave=False, desc=f"Epoch {episode+1}"):
            # Получение и предобработка текущего состояния
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
            
            # Выбор действия
            action_distribution, action_log_prob = self.agent.get_action(state)
            
            # Для дискретного набора действий, округляем до ближайшего индекса
            if self.actions is not None:
                # В случае дискретных действий
                action_distribution = torch.Tensor(action_distribution)
                selected_action_idx = int(torch.argmax(action_distribution).item())
                selected_action = self.actions[selected_action_idx]
            else:
                # Для непрерывных действий
                selected_action = action_distribution.detach().cpu().numpy()
            
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
            intrinsic_reward = self.agent.compute_intrinsic_reward(state, selected_action_idx, next_state)
            total_intrinsic += intrinsic_reward
            combined_reward = reward + self.agent.lambda_intrinsic * intrinsic_reward
            
            # Сохранение перехода в память агента и буфер воспроизведения
            self.agent.append_memory(state, selected_action_idx, next_state, reward, combined_reward, action_log_prob, done)
            
            # Логирование данных (например, отправка в Kafka)
            publish_data(
                array=temporal_state, 
                epoch=episode, 
                loss=0.0,
                mean_reward=np.array(train_scores).mean() if train_scores else 0.0, 
                mode="Train"
            )
            
            global_step += 1
            
            # Обучение агента, если буфер заполнен
            if global_step > self.agent.batch_size and len(self.agent.memory) >= self.agent.batch_size:
                policy_loss, value_loss = self.agent.train_agent()
                loss_lst.append((policy_loss, value_loss))
            
            # Завершение эпизода
            if done:
                total_episode_reward = self.env.get_total_reward()
                train_scores.append(total_episode_reward)
                self.env.new_episode()
                break
        
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
        
        return total_reward, np.array(loss_lst)



    def save_model(self, path: str) -> None:
        """
        Сохранение моделей агента на диск.

        Args:
            path (str): Базовый путь для сохранения моделей (без расширений).
        """
        torch.save({
            'policy_net_state_dict': self.agent.policy_net.state_dict(),
            'value_net_state_dict': self.agent.value_net.state_dict(),
            'forward_model_state_dict': self.agent.forward_model.state_dict()
        }, f"{path}.pth")
        print(f"Models saved to {path}.pth")

    def load_model(self, path: str) -> None:
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
        
        print(f"Models and optimizers loaded from {path}.pth")


