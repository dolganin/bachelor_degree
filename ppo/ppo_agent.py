import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Bernoulli
import numpy as np
import torch.nn.functional as F

from ppo.pnetwork import PolicyNetwork
from ppo.vnetwork import ValueNetwork
from ppo.shared_transformer import SharedTransformer
from base.agent_base import RLAgent

class PPOAgent(RLAgent):
    def __init__(self, 
                 action_size: int,
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
        
        
    def get_action(self, state: np.ndarray):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.policy_net.eval()
        with torch.no_grad():
            logits = self.policy_net(state)  # (1, action_size)
            probs = torch.sigmoid(logits)
            dist = Bernoulli(probs)
            action = dist.sample()
            action_log_prob = dist.log_prob(action).sum(dim=-1)
        self.policy_net.train()
        return action.cpu().numpy()[0], action_log_prob.cpu()
    
    def train_agent(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, 
                    next_states: torch.Tensor, log_probs: torch.Tensor, dones: torch.Tensor):
        """
        Выполняет PPO-обновление на переданном батче данных.

        Args:
            states (torch.Tensor): Тензор состояний (batch_size, ...).
            actions (torch.Tensor): Тензор действий (batch_size, action_size), бинарный.
            rewards (torch.Tensor): Тензор наград (batch_size,).
            next_states (torch.Tensor): Тензор следующих состояний (batch_size, ...).
            log_probs (torch.Tensor): Тензор логарифмов вероятностей (batch_size,).
            dones (torch.Tensor): Тензор флагов завершения эпизода (batch_size,).

        Returns:
            tuple: (policy_loss, value_loss, diagnostics)
        """
        # Перенос данных на устройство
        states = states.to(self.device)
        actions = actions.to(self.device)
        next_states = next_states.to(self.device)
        rewards = rewards.to(self.device)
        log_probs = log_probs.to(self.device)
        dones = dones.to(self.device)

        # Значения и returns через Value Network
        values = self.value_net(states).squeeze()
        next_values = self.value_net(next_states).squeeze()
        returns = rewards + self.discount_factor * next_values * (1 - dones.float())
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Обновление Value Network
        value_loss = F.smooth_l1_loss(values, returns)
        
        # Обновление Policy Network
        logits = self.policy_net(states)
        probs = torch.sigmoid(logits)
        dist = Bernoulli(probs)

        # Убедимся, что actions бинарные и имеют форму (batch, action_size)
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1).expand_as(probs)

        new_log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()
        ratio = torch.exp(new_log_probs - log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

        # Общая потеря и backward
        total_loss = policy_loss + value_loss

        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        total_loss.backward()

        # Нормы градиентов
        policy_grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.policy_net.parameters() if p.grad is not None) ** 0.5
        value_grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.value_net.parameters() if p.grad is not None) ** 0.5

        self.policy_optimizer.step()
        self.value_optimizer.step()

        diagnostics = {
            'entropy': entropy.item(),
            'policy_grad_norm': policy_grad_norm,
            'value_grad_norm': value_grad_norm,
            'advantages_mean': advantages.mean().item(),
            'advantages_std': advantages.std().item(),
            'action_distribution_mean': probs.mean().item(),
            'action_distribution_std': probs.std().item()
        }
        
        return policy_loss.item(), value_loss.item(), diagnostics


    def append_memory(self, state, action, reward, next_state, done):
        pass

    def update_target_net(self):
        pass
