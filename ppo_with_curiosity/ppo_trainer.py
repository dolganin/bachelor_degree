import torch
import numpy as np
from tqdm import trange
from torch.nn import Module
from base.trainer_base import TrainerRL
from utilities.preprocessing import preprocess
from server_consumer.broker_kafka import publish_data
from utilities.video_logger import VideoLogger
from ppo_with_curiosity.ppo_agent import PPOAgent
from .replay_buffer import ReplayBuffer
import cv2
from base.agent_evaluator import AgentEvaluator

class PPOTrainer(TrainerRL):
    def __init__(self, env, agent: Module, video_logger: VideoLogger=None, wandb_logger=None,  
                 device: str = "cpu", resolution: tuple = (30, 45), frame_repeat: int = 45, 
                 steps_per_epoch: int = 1000, actions: list = None, test_episodes_per_epoch: int = 1000, 
                 model_savefile: str = None, buffer_capacity: int = 10000, buffer_momentum: float = 0.995, 
                 agent_evaluator: AgentEvaluator = None):
        super(PPOTrainer, self).__init__()
        self.env = env
        self.agent = agent
        self.current_step = 0
        self.total_rewards = []
        self.video_logger = video_logger
        self.wandb_logger = wandb_logger
        self.device = device
        self.resolution = resolution
        self.frame_repeat = frame_repeat
        self.steps_per_epoch = steps_per_epoch
        self.actions = actions
        self.test_episodes_per_epoch = test_episodes_per_epoch
        self.model_savefile = model_savefile
        self.memory = ReplayBuffer(capacity=buffer_capacity, momentum=buffer_momentum)
        self.avaluator = agent_evaluator

    def train(self, episode: int, steps_per_epoch: int = 1000):
        loss_dict = {}
        self.env.new_episode()
        train_scores = []
        global_step = 0
        total_reward = 0.0
        total_intrinsic = 0.0

        average_value_loss = 0.0
        average_policy_loss = 0.0
        average_forward_loss = 0.0

        # Используем tqdm для создания прогресс-бара
        with trange(steps_per_epoch, desc=f"Epoch {episode}", unit="step") as t:
            for _ in t:
                # Получение и предобработка текущего состояния
                raw_state = self.env.get_state().screen_buffer
                state = preprocess(raw_state, resolution=self.resolution)

                # Логирование видеофрейма
                temporal_state = np.array(raw_state, dtype=np.uint8)
                if temporal_state.shape[-1] == 3:
                    temporal_state = temporal_state[..., ::-1]
                temporal_state = cv2.resize(temporal_state, (1280, 720), interpolation=cv2.INTER_LINEAR)

                publish_data(
                    array=temporal_state, 
                    epoch=episode, 
                    loss=np.mean([average_policy_loss, average_value_loss, average_forward_loss]),
                    mean_reward=np.array(train_scores).mean() if train_scores else 0.0, 
                    mode="Train"
                )
                
                # Выбор действия
                action_distribution, action_log_prob = self.agent.get_action(state)
                if self.actions is not None:
                    action_distribution = torch.Tensor(action_distribution)
                    selected_action_idx = int(torch.argmax(action_distribution).item())
                    selected_action = self.actions[selected_action_idx]
                else:
                    selected_action = action_distribution.detach().cpu().numpy()

                # Выполнение действия в среде
                reward = self.env.make_action(selected_action, self.frame_repeat)
                done = self.env.is_episode_finished()
                total_reward += reward

                if not done:
                    next_raw_state = self.env.get_state().screen_buffer
                    next_state = preprocess(next_raw_state, resolution=self.resolution)
                else:
                    next_state = np.zeros((3, self.resolution[0], self.resolution[1]), dtype=np.float32)

                intrinsic_reward = self.agent.compute_intrinsic_reward(state, action_distribution, next_state)
                total_intrinsic += intrinsic_reward
                combined_reward = reward + self.agent.lambda_intrinsic * intrinsic_reward
                self.agent.append_memory(state, action_distribution, next_state, reward, combined_reward, action_log_prob, done)

                global_step += 1

                # Обучение агента, если буфер заполнен
                if global_step > self.agent.batch_size // 10 and len(self.agent.forward_replay_buffer) >= self.agent.batch_size:
                    policy_loss, value_loss = self.agent.train_agent()
                    self.wandb_logger.log({
                        'Train policiy loss': policy_loss,
                        'Train value loss': value_loss,
                    })
                    loss_dict['policy_loss'] = loss_dict.get('policy_loss', []) + [policy_loss]
                    loss_dict['value_loss'] = loss_dict.get('value_loss', []) + [value_loss]
                else:
                    loss_dict['policy_loss'] = loss_dict.get('policy_loss', []) + [0.0]
                    loss_dict['value_loss'] = loss_dict.get('value_loss', []) + [0.0]

                if done:
                    total_episode_reward = self.env.get_total_reward()
                    train_scores.append(total_episode_reward)
                    self.env.new_episode()    

                # Обновляем значения tqdm
                t.set_postfix({
                    "Reward": f"{total_reward:.2f}",
                    "Intrinsic": f"{total_intrinsic:.2f}",
                    "Policy Loss": f"{np.mean(loss_dict['policy_loss']):.4f}",
                    "Value Loss": f"{np.mean(loss_dict['value_loss']):.4f}",
                    "Forward Loss": f"{np.mean(loss_dict.get('forward_loss', [0.0])):.4f}"
                })

        forward_loss = self.agent.update_forward_model()
        self.wandb_logger.log({"Forward Loss": forward_loss})
        loss_dict['forward_loss'] = loss_dict.get('forward_loss', []) + [forward_loss] if forward_loss > 0.0 else [0.0]
        
        average_policy_loss = np.mean(loss_dict['policy_loss']) if 'policy_loss' in loss_dict else 0.0
        average_value_loss = np.mean(loss_dict['value_loss']) if 'value_loss' in loss_dict else 0.0
        average_forward_loss = np.mean(loss_dict['forward_loss']) if 'forward_loss' in loss_dict else 0.0

        
        return total_reward, loss_dict
    


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
        self.agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.agent.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.agent.forward_model.load_state_dict(checkpoint['forward_model_state_dict'])
        
        self.agent.policy_net.to(self.device)
        self.agent.value_net.to(self.device)
        self.agent.forward_model.to(self.device)
        
        print(f"Models and optimizers loaded from {path}")


