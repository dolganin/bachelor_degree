# train.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from torch.distributions import Normal
from collections import deque
import random

from ppo_with_curiosity.pnetwork import PolicyNetwork
from ppo_with_curiosity.vnetwork import ValueNetwork
from ppo_with_curiosity.forward_model import ForwardModelCNN
from ppo_with_curiosity.shared_transformer import SharedTransformer
from ppo_with_curiosity.replay_buffer import ReplayBuffer  # Импорт ReplayBuffer из replay_buffer.py

def select_action(state, policy_net):
    """
    Выбирает действие на основе текущего состояния и политики агента.
    
    Args:
        state (np.ndarray): Текущее состояние среды.
        policy_net (PolicyNetwork): Сеть политики.
    
    Returns:
        tuple: Выбранное действие и логарифм вероятности действия.
    """
    state = torch.FloatTensor(state).unsqueeze(0)  # (1, C, H, W)
    mean, std = policy_net(state)
    dist = Normal(mean, std)
    action = dist.sample()
    action_log_prob = dist.log_prob(action).sum(dim=-1)
    return action.detach().numpy()[0], action_log_prob.detach()

def compute_intrinsic_reward(state, action, next_state, forward_model):
    """
    Вычисляет внутреннее вознаграждение на основе предсказательной ошибки Forward Model.
    
    Args:
        state (np.ndarray): Текущее состояние.
        action (np.ndarray): Действие агента.
        next_state (np.ndarray): Следующее состояние после действия.
        forward_model (ForwardModelCNN): Модель предсказания следующего состояния.
    
    Returns:
        float: Внутреннее вознаграждение.
    """
    state_tensor = torch.FloatTensor(state).unsqueeze(0)  # (1, C, H, W)
    action_tensor = torch.FloatTensor(action).unsqueeze(0)  # (1, action_dim)
    next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)  # (1, C, H, W)
    predicted_next_state = forward_model(state_tensor, action_tensor)
    intrinsic_reward = (predicted_next_state - next_state_tensor).pow(2).mean().item()
    return intrinsic_reward

def train():
    # Параметры обучения
    num_episodes = 1000
    gamma = 0.99
    lambda_intrinsic = 0.1  # Вес внутреннего вознаграждения
    batch_size = 64
    replay_buffer_capacity = 10000
    
    # Инициализация среды
    env = gym.make('Pendulum-v1')  # Замените на вашу среду
    image_channels = env.observation_space.shape[0]  # Например, 3 для RGB
    image_size = env.observation_space.shape[1]  # Предполагается квадратное изображение
    action_dim = env.action_space.shape[0]  # Для непрерывных действий
    
    # Инициализация моделей
    shared_transformer = SharedTransformer(
        image_channels=image_channels,
        image_size=image_size,
        patch_size=7,
        embedding_dim=128,
        num_heads=8,
        num_layers=6,
        mlp_dim=256,
        dropout=0.1
    )
    
    policy_net = PolicyNetwork(shared_transformer, action_dim=action_dim)
    value_net = ValueNetwork(shared_transformer)
    forward_model = ForwardModelCNN(action_dim=action_dim, image_channels=image_channels, image_size=image_size, hidden_dim=128)
    
    # Инициализация оптимизаторов
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=3e-4)
    value_optimizer = optim.Adam(value_net.parameters(), lr=1e-3)
    forward_optimizer = optim.Adam(forward_model.parameters(), lr=1e-3)
    
    # Инициализация Replay Buffer
    replay_buffer = ReplayBuffer(capacity=replay_buffer_capacity)
    
    # Перемещение моделей на доступное устройство (GPU или CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net.to(device)
    value_net.to(device)
    forward_model.to(device)
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        trajectory = []
        total_reward = 0
        total_intrinsic = 0
    
        while not done:
            # Выбор действия
            action, action_log_prob = select_action(state, policy_net)
            action_clipped = np.clip(action, env.action_space.low, env.action_space.high)  # Клиппинг действия
            next_state, reward, done, _ = env.step(action_clipped)
            
            # Вычисление внутреннего вознаграждения
            intrinsic_reward = compute_intrinsic_reward(state, action, next_state, forward_model)
            
            # Общее вознаграждение
            total_reward += reward
            total_intrinsic += intrinsic_reward
            combined_reward = reward + lambda_intrinsic * intrinsic_reward
            
            # Сохранение перехода
            trajectory.append((state, action, combined_reward, action_log_prob, next_state))
            
            # Добавление в Replay Buffer
            replay_buffer.push(state, action, next_state)
            
            state = next_state
    
        # Обработка траектории
        states, actions, rewards, log_probs, next_states = zip(*trajectory)
        states = torch.FloatTensor(states).to(device)  # (batch_size, C, H, W)
        actions = torch.FloatTensor(actions).to(device)  # (batch_size, action_dim)
        rewards = torch.FloatTensor(rewards).to(device)  # (batch_size,)
        log_probs = torch.stack(log_probs).to(device)  # (batch_size,)
        next_states = torch.FloatTensor(next_states).to(device)  # (batch_size, C, H, W)
    
        # Вычисление дисконтированных вознаграждений
        discounted_rewards = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            discounted_rewards.insert(0, R)
        discounted_rewards = torch.FloatTensor(discounted_rewards).to(device)
        advantages = discounted_rewards - value_net(states).squeeze()
    
        # Обновление сети ценности
        value_loss = nn.MSELoss()(value_net(states).squeeze(), discounted_rewards)
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()
    
        # Обновление сети политики (PPO-часть)
        ratio = torch.exp(log_probs - log_probs.detach())
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
    
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()
    
        # Обновление Forward Model (через Replay Buffer)
        if len(replay_buffer) >= batch_size:
            states_batch, actions_batch, next_states_batch = replay_buffer.sample(batch_size)
            states_batch = states_batch.to(device)
            actions_batch = actions_batch.to(device)
            next_states_batch = next_states_batch.to(device)
            
            predicted_next_states = forward_model(states_batch, actions_batch)
            forward_loss = nn.MSELoss()(predicted_next_states, next_states_batch)
            forward_optimizer.zero_grad()
            forward_loss.backward()
            forward_optimizer.step()
        else:
            forward_loss = torch.tensor(0.0).to(device)
    
        # Логирование прогресса
        print(f"Episode {episode+1}/{num_episodes}, Reward: {total_reward:.2f}, Intrinsic: {total_intrinsic:.2f}, Forward Loss: {forward_loss.item():.4f}")
    
        # Сохранение моделей каждые N эпизодов
        if (episode + 1) % 100 == 0:
            torch.save(policy_net.state_dict(), f"policy_net_episode_{episode+1}.pth")
            torch.save(value_net.state_dict(), f"value_net_episode_{episode+1}.pth")
            torch.save(forward_model.state_dict(), f"forward_model_episode_{episode+1}.pth")
