import torch
import numpy as np
from tqdm import trange
from torch.nn import Module
from base.trainer_base import TrainerRL
from utilities.preprocessing import preprocess
from server_consumer.broker_kafka import publish_data
from utilities.video_logger import VideoLogger
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
        self.avaluator = agent_evaluator

    def train(self, episode: int, steps_per_epoch: int = 1000):
        loss_dict = {}
        self.env.new_episode()
        train_scores = []
        episode_rewards = []  # Для скользящего среднего награды за 100 эпизодов
        total_reward = 0.0
        global_step = 0

        # Собираем траекторию онлайн за эпоху
        trajectories = {
            'states': [],
            'actions': [],
            'rewards': [],
            'log_probs': [],
            'next_states': [],
            'dones': []
        }

        with trange(steps_per_epoch, desc=f"Epoch {episode}", unit="step") as t:
            for _ in t:
                raw_state = self.env.get_state().screen_buffer
                state = preprocess(raw_state, resolution=self.resolution)

                # Выбор действия
                action, action_log_prob = self.agent.get_action(state)
                if self.actions is not None:
                    action_tensor = torch.Tensor(action)
                    selected_action_idx = int(torch.argmax(action_tensor).item())
                    selected_action = self.actions[selected_action_idx]
                else:
                    selected_action = action

                reward = self.env.make_action(selected_action, self.frame_repeat)
                done = self.env.is_episode_finished()
                total_reward += reward

                if not done:
                    next_raw_state = self.env.get_state().screen_buffer
                    next_state = preprocess(next_raw_state, resolution=self.resolution)
                else:
                    next_state = np.zeros((3, self.resolution[0], self.resolution[1]), dtype=np.float32)

                # Сохраняем траекторию
                trajectories['states'].append(state)
                trajectories['actions'].append(action)
                trajectories['rewards'].append(reward)
                trajectories['log_probs'].append(action_log_prob)
                trajectories['next_states'].append(next_state)
                trajectories['dones'].append(float(done))

                global_step += 1
                self.wandb_logger.log({'Reward': reward})

                if done:
                    ep_reward = self.env.get_total_reward()
                    self.wandb_logger.log({'Episode Reward': ep_reward})
                    train_scores.append(ep_reward)
                    episode_rewards.append(ep_reward)
                    self.env.new_episode()

                t.set_postfix({"Reward": f"{reward:.2f}"})

            # Преобразуем списки в тензоры
            states_tensor = torch.FloatTensor(np.array(trajectories['states']))
            actions_tensor = torch.tensor(np.array(trajectories['actions']))
            rewards_tensor = torch.FloatTensor(np.array(trajectories['rewards']))
            log_probs_tensor = torch.cat(trajectories['log_probs'])
            next_states_tensor = torch.FloatTensor(np.array(trajectories['next_states']))
            dones_tensor = torch.FloatTensor(np.array(trajectories['dones']))

            # Обновляем сеть с онлайн траекторией
            policy_loss, value_loss, diagnostics = self.agent.train_agent(
                states_tensor, actions_tensor, rewards_tensor, next_states_tensor, log_probs_tensor, dones_tensor
            )
            loss_dict.setdefault('policy_loss', []).append(policy_loss)
            loss_dict.setdefault('value_loss', []).append(value_loss)

            self.wandb_logger.log({
                'Train policy loss': policy_loss,
                'Train value loss': value_loss,
                'Policy Entropy': diagnostics.get('entropy', 0.0),
                'Policy Grad Norm': diagnostics.get('policy_grad_norm', 0.0),
                'Value Grad Norm': diagnostics.get('value_grad_norm', 0.0),
                'Advantages Mean': diagnostics.get('advantages_mean', 0.0),
                'Advantages Std': diagnostics.get('advantages_std', 0.0),
                'Action Distribution Mean': diagnostics.get('action_distribution_mean', 0.0),
                'Action Distribution Std': diagnostics.get('action_distribution_std', 0.0)
            })

            if len(episode_rewards) >= 100:
                avg_reward = np.mean(episode_rewards[-100:])
                self.wandb_logger.log({'Average Train Reward (last 100 episodes)': avg_reward})

        return total_reward, loss_dict


    
    def save_model(self, path: str) -> None:
        torch.save({
            'policy_net_state_dict': self.agent.policy_net.state_dict(),
            'value_net_state_dict': self.agent.value_net.state_dict()
        }, f"{path}.pth")
        print(f"Models saved to {path}.pth")
    
    def load_model(self, path: str) -> None:
        checkpoint = torch.load(f"{path}.pth", map_location=self.device)
        self.agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.agent.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.agent.policy_net.to(self.device)
        self.agent.value_net.to(self.device)
        print(f"Models and optimizers loaded from {path}")
