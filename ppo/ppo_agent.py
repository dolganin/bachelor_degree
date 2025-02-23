import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Normal
import numpy as np
import torch.nn.functional as F

from ppo.pnetwork import PolicyNetwork
from ppo.vnetwork import ValueNetwork
from ppo.shared_transformer import SharedTransformer
from ppo.replay_buffer import ReplayBuffer
from base.agent_base import RLAgent

class PPOAgent(RLAgent):
    def __init__(self, 
                 action_size: int,
                 memory_size: int,
                 batch_size: int,
                 discount_factor: float,
                 lr_value: float,
                 lr_policy: float,
                 device: torch.device,
                 entropy_coef: float = 0.01,
                 clip_epsilon: float = 0.2,
                 screen_resolution: tuple = None,
                 channels: int = 3,
                 patch_size: int = 10,
                 dropout_rate: float = 0.1,
                 embedding_dim: int = 120,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 mlp_dim: int = 256
                 ):
        
        self.entropy_coef = entropy_coef
        self.clip_epsilon = clip_epsilon
        self.action_size = action_size
        self.device = device
        self.memory_size = memory_size
        self.batch_size = batch_size
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
        
        # Инициализация моделей
        self.shared_transformer = SharedTransformer(
            image_channels=self.channels,
            image_height=self.screen_resolution[0],
            image_width=self.screen_resolution[1],
            patch_size=self.patch_size,
            embedding_dim=self.embedding_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            mlp_dim=self.mlp_dim,
            dropout=self.dropout_rate
        ).to(self.device)
        
        self.policy_net = PolicyNetwork(self.shared_transformer, action_dim=self.action_size).to(self.device)
        self.value_net = ValueNetwork(self.shared_transformer).to(self.device)
        
        # Инициализация оптимизаторов
        self.policy_optimizer = Adam(self.policy_net.parameters(), lr=self.lr_policy)
        self.value_optimizer = Adam(self.value_net.parameters(), lr=self.lr_value)
        
        # Инициализация Replay Buffer
        self.replay_buffer = ReplayBuffer(capacity=self.memory_size)
        
    def get_action(self, state: np.ndarray):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mean, std = self.policy_net(state)
        std = torch.clamp(std, min=1e-6, max=1.0)
        #print(mean, std)
        dist = Normal(mean, std)
        action = dist.sample()
        action_log_prob = dist.log_prob(action).sum(dim=-1)
        return action.detach().cpu().numpy()[0], action_log_prob.detach()
    
    def train_agent(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0, {}

        # Получаем батч из памяти
        states, actions, next_states, rewards, log_probs, dones = \
            self.replay_buffer.sample(self.batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        next_states = next_states.to(self.device)
        rewards = rewards.to(self.device)
        log_probs = log_probs.to(self.device)
        dones = dones.to(self.device)

        values = self.value_net(states).squeeze()
        next_values = self.value_net(next_states).squeeze()
        advantages = (rewards + self.discount_factor * next_values * (1 - dones.float()) - values).detach()
        returns = rewards + self.discount_factor * next_values * (1 - dones.float())

        # Обновление Value Network
        value_loss = F.smooth_l1_loss(values, returns)
        self.value_optimizer.zero_grad()
        value_loss.backward(retain_graph=True)
        # Вычисляем норму градиентов для value_net
        value_grad_norm = 0.0
        for p in self.value_net.parameters():
            if p.grad is not None:
                value_grad_norm += p.grad.data.norm(2).item() ** 2
        value_grad_norm = value_grad_norm ** 0.5
        self.value_optimizer.step()

        # Обновление Policy Network
        self.policy_optimizer.zero_grad()
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

        policy_loss.backward()
        # Вычисляем норму градиентов для policy_net
        policy_grad_norm = 0.0
        for p in self.policy_net.parameters():
            if p.grad is not None:
                policy_grad_norm += p.grad.data.norm(2).item() ** 2
        policy_grad_norm = policy_grad_norm ** 0.5
        self.policy_optimizer.step()

        diagnostics = {
            'entropy': entropy.item(),
            'policy_grad_norm': policy_grad_norm,
            'value_grad_norm': value_grad_norm,
            'advantages_mean': advantages.mean().item(),
            'advantages_std': advantages.std().item(),
            'action_distribution_mean': mean.mean().item(),
            'action_distribution_std': mean.std().item()
        }

        return policy_loss.item(), value_loss.item(), diagnostics

    
    def append_memory(self, state: np.ndarray, action: int, next_state: np.ndarray, reward: float, action_log_prob: np.ndarray, done: bool):
        self.replay_buffer.push(state=state, action=action, next_state=next_state, reward=reward, action_log_prob=action_log_prob, done=done)

    def update_target_net(self):
        pass


    def compute_total_loss(self):
        """
        Вычисляет общий лосс, складывая потери от Policy Network, Value Network и Forward Model.

        Returns:
            float: Общий лосс.
        """
        # Обучение агента и получение потерь
        policy_loss, value_loss, _ = self.train_agent()
        # Общий лосс
        total_loss = policy_loss + value_loss
        return total_loss