# test.py

import torch
import numpy as np
import gym

from ppo_with_curiosity.pnetwork import PolicyNetwork
from ppo_with_curiosity.shared_transformer import SharedTransformer

def select_action(state, policy_net, device):
    """
    Выбирает действие на основе текущего состояния и политики агента.
    
    Args:
        state (np.ndarray): Текущее состояние среды.
        policy_net (PolicyNetwork): Сеть политики.
        device (torch.device): Устройство (CPU или GPU).
    
    Returns:
        np.ndarray: Выбранное действие.
    """
    state = torch.FloatTensor(state).unsqueeze(0).to(device)  # (1, C, H, W)
    mean, std = policy_net(state)
    dist = torch.distributions.Normal(mean, std)
    action = dist.mean  # Используем среднее как детерминированное действие
    action_clipped = torch.clamp(action, -1.0, 1.0)  # Клиппинг действия в диапазон среды
    return action_clipped.detach().cpu().numpy()[0]

def test():
    # Параметры
    num_test_episodes = 100
    render = False  # Установите True, если хотите визуализировать среду
    
    # Инициализация среды
    env = gym.make('Pendulum-v1')  # Замените на вашу среду
    image_channels = env.observation_space.shape[0]
    image_size = env.observation_space.shape[1]
    action_dim = env.action_space.shape[0]
    
    # Инициализация Policy Network
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
    
    # Загрузка сохранённых весов модели
    trained_policy_path = "policy_net_episode_1000.pth"  # Замените на путь к вашей сохранённой модели
    policy_net.load_state_dict(torch.load(trained_policy_path, map_location=torch.device('cpu')))
    policy_net.eval()
    
    # Перемещение модели на устройство
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net.to(device)
    
    total_rewards = []
    
    for episode in range(num_test_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            if render:
                env.render()
            
            # Выбор действия
            action = select_action(state, policy_net, device)
            action_clipped = np.clip(action, env.action_space.low, env.action_space.high)
            
            # Шаг в среде
            next_state, reward, done, _ = env.step(action_clipped)
            
            episode_reward += reward
            state = next_state
        
        total_rewards.append(episode_reward)
        print(f"Test Episode {episode+1}/{num_test_episodes}, Reward: {episode_reward:.2f}")
    
    env.close()
    
    average_reward = np.mean(total_rewards)
    print(f"Average Reward over {num_test_episodes} episodes: {average_reward:.2f}")

