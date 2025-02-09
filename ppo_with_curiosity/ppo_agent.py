# PPO_Agent.py

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Normal
import numpy as np
import torch.nn.functional as F  # Import F for Huber loss

from ppo_with_curiosity.pnetwork import PolicyNetwork
from ppo_with_curiosity.vnetwork import ValueNetwork
from ppo_with_curiosity.forward_model import ForwardModelCNN
from ppo_with_curiosity.shared_transformer import SharedTransformer
from ppo_with_curiosity.replay_buffer import ReplayBuffer
from base.agent_base import RLAgent

class PPOAgent(RLAgent):
    def __init__(self, 
                 action_size: int,
                 memory_size: int,
                 batch_size: int,
                 discount_factor: float,
                 lr_forward: float,
                 lr_value: float,
                 lr_policy: float,
                 device: torch.device,
                 lambda_intrinsic: float = 0.1,
                 entropy_coef: float = 0.01,
                 clip_epsilon: float = 0.2,
                 hidden_dim: int = 128,
                 screen_resolution: tuple = None,
                 channels: int = 3,
                 patch_size: int = 10,
                 dropout_rate: float = 0.1,
                 embedding_dim: int = 120,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 mlp_dim: int = 256,
                 ex_loss: float = 0.0
                 ):
        """
        Инициализация PPOAgent с настройками для PPO with Curiosity.

        Args:
            action_size (int): Количество возможных действий.
            memory_size (int): Размер буфера памяти.
            batch_size (int): Размер батча для обучения.
            discount_factor (float): Коэффициент дисконтирования.
            lr (float): Скорость обучения.
            device (torch.device): Устройство для вычислений.
            model_savefile (str): Путь для сохранения модели.
            lambda_intrinsic (float, optional): Вес внутреннего вознаграждения. По умолчанию 0.1.
            entropy_coef (float, optional): Коэффициент энтропии для регуляризации. По умолчанию 0.01.
            clip_epsilon (float, optional): Коэффициент для ограничения обновлений в PPO. По умолчанию 0.2.
            hidden_dim (int, optional): Размер скрытого слоя для Forward Model. По умолчанию 128.
        """
        
        self.lambda_intrinsic = lambda_intrinsic
        self.entropy_coef = entropy_coef
        self.clip_epsilon = clip_epsilon
        self.hidden_dim = hidden_dim
        self.action_size = action_size
        self.device = device
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.lr_forward = lr_forward
        self.lr_value = lr_value
        self.lr_policy = lr_policy
        
        self.discount_factor = discount_factor
        self.screen_resolution = screen_resolution
        self.channels = channels
        self.patch_size = patch_size
        self.dropout_rate = dropout_rate
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_dim = mlp_dim
        self.ex_loss = ex_loss
        
        # Инициализация моделей
        self.shared_transformer = SharedTransformer(
            image_channels=self.channels,  # Предполагается RGB; изменить при необходимости
            image_height=self.screen_resolution[0],
            image_width=self.screen_resolution[1],       # Измените в соответствии с вашей средой
            patch_size=self.patch_size,
            embedding_dim=self.embedding_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            mlp_dim=self.mlp_dim,
            dropout=self.dropout_rate
        ).to(self.device)
        
        self.policy_net = PolicyNetwork(self.shared_transformer, action_dim=self.action_size).to(self.device)
        self.value_net = ValueNetwork(self.shared_transformer).to(self.device)
        self.forward_model = ForwardModelCNN(
            action_dim=self.action_size,
            image_channels=self.channels,  # Предполагается RGB; изменить при необходимости
            input_height=self.screen_resolution[0],
            input_width=self.screen_resolution[1],      # Измените в соответствии с вашей средой
            hidden_dim=self.hidden_dim
        ).to(self.device)
        
        # Инициализация оптимизаторов
        self.policy_optimizer = Adam(self.policy_net.parameters(), lr=self.lr_policy)
        self.value_optimizer = Adam(self.value_net.parameters(), lr=self.lr_value)
        self.forward_optimizer = Adam(self.forward_model.parameters(), lr=self.lr_forward)
        
        # Инициализация Replay Buffer для Forward Model
        self.forward_replay_buffer = ReplayBuffer(capacity=self.memory_size)
        
    def get_action(self, state: np.ndarray):
        """
        Выбор действия с защитой от NaN в Policy Network.
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # (1, C, H, W)
        mean, std = self.policy_net(state)
        
        # Ограничение std для предотвращения деления на ноль или числовой нестабильности
        std = torch.clamp(std, min=1e-6, max=1.0)  # минимизация std для предотвращения деления на ноль

        dist = Normal(mean, std)

        # Проверка на NaN в mean и std
        if torch.any(torch.isnan(mean)) or torch.any(torch.isnan(std)):
            print(f"Warning: NaN detected in policy network output! mean: {mean}, std: {std}")
            return np.zeros(self.action_size), torch.zeros_like(mean)

        action = dist.sample()
        action_log_prob = dist.log_prob(action).sum(dim=-1)

        return action.detach().cpu().numpy()[0], action_log_prob.detach()


    def compute_intrinsic_reward(self, state: np.ndarray, action: int, next_state: np.ndarray):
        """
        Вычисляет внутреннее вознаграждение на основе предсказательной ошибки Forward Model.

        Args:
            state (np.ndarray): Текущее состояние.
            action (int): Действие агента.
            next_state (np.ndarray): Следующее состояние после действия.

        Returns:
            float: Внутреннее вознаграждение.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # (1, C, H, W)
        action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)  # (1, action_dim)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)  # (1, C, H, W)
        with torch.no_grad():
            predicted_next_state = self.forward_model(state_tensor, action_tensor)
        intrinsic_reward = (predicted_next_state - next_state_tensor).pow(2).mean().item()
        return intrinsic_reward

    def train_agent(self):
        if len(self.forward_replay_buffer) < self.batch_size:
            return self.ex_loss, self.ex_loss

        states, actions, next_states, rewards, combined_rewards, log_probs, dones = \
            self.forward_replay_buffer.sample(self.batch_size)

        states, actions, next_states, rewards, combined_rewards, log_probs, dones = \
            states.to(self.device), actions.to(self.device), next_states.to(self.device), \
            rewards.to(self.device), combined_rewards.to(self.device), log_probs.to(self.device), dones.to(self.device)

        values = self.value_net(states).squeeze()
        next_values = self.value_net(next_states).squeeze()
        advantages = (combined_rewards + self.discount_factor * next_values * (1 - dones.float()) - values).detach()
        returns = combined_rewards + self.discount_factor * next_values * (1 - dones.float())

        # Update Value Network using Huber loss
        value_loss = F.smooth_l1_loss(values, returns)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Обновление Policy Network
        mean, std = self.policy_net(states)

        std = torch.clamp(std, min=1e-6, max=1.0)

        dist = Normal(mean, std)

        if actions.dim() == 1:
            actions = actions.unsqueeze(-1).expand_as(mean)

        new_log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()

        ratio = torch.exp(new_log_probs - log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

        if torch.isnan(policy_loss).any():
            print(f"Warning: NaN detected in policy loss calculation!")

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        return policy_loss.item(), value_loss.item()

    def update_forward_model(self):
        if len(self.forward_replay_buffer) < self.batch_size:
            return 0.0  # Not enough data to train the forward model

        states, actions, next_states, rewards, combined_rewards, log_probs, dones = \
            self.forward_replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)         # (batch_size, C, H, W)
        actions = torch.FloatTensor(actions).to(self.device)       # (batch_size, action_dim)
        next_states = torch.FloatTensor(next_states).to(self.device) # (batch_size, C, H, W)
        
        predicted_next_states = self.forward_model(states, actions)
        
        # Use Huber loss for the forward model
        forward_loss = F.smooth_l1_loss(predicted_next_states, next_states)
        
        self.forward_optimizer.zero_grad()
        forward_loss.backward()
        self.forward_optimizer.step()
        
        return forward_loss.item()

    def update_target_net(self):
        """
        Метод оставлен пустым, так как в PPO обычно не используется целевая сеть.
        """
        pass

    def append_memory(self,  state: np.ndarray, action: int, next_state: np.ndarray, reward: float, \
                      combined_reward: float, action_log_prob: np.ndarray, done: bool) -> None:
        """
        Добавлеие в память модели предыдущего состояния для стабилизации обучения (впервые применено Minh. et al 2015 в Atari)
        """

        self.forward_replay_buffer.push(state, action, next_state, combined_reward, action_log_prob, reward, done)

    def compute_total_loss(self):
        """
        Вычисляет общий лосс, складывая потери от Policy Network, Value Network и Forward Model.

        Returns:
            float: Общий лосс.
        """
        # Обучение агента и получение потерь
        policy_loss, value_loss = self.train_agent()
        forward_loss = self.update_forward_model()

        # Общий лосс
        total_loss = policy_loss + value_loss + forward_loss
        return total_loss