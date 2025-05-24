import torch
import numpy as np
from tqdm import trange
from torch.nn import Module
from base.trainer_base import TrainerRL
from utilities.preprocessing import preprocess
from server_consumer.broker_kafka import publish_data
from utilities.video_logger import VideoLogger
from base.agent_evaluator import AgentEvaluator

class PPOTrainer(TrainerRL):
    def __init__(self, env, agent: Module, video_logger: VideoLogger=None, wandb_logger=None,  
                 device: str = "cpu", resolution: tuple = (30, 45), frame_repeat: int = 45, actions: list = None,  
                 model_savefile: str = None, agent_evaluator: AgentEvaluator = None, ppo_epochs: int = 5):
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
        self.actions = actions
        self.model_savefile = model_savefile
        self.avaluator = agent_evaluator
        self.ppo_epochs = ppo_epochs

    def train(self, total_steps: int, batch_size: int = 64):
        loss_dict = {}
        self.env.new_episode()
        episode_rewards = []
        total_reward = 0.0
        step = 0

        trajectories = {
            'states': [],
            'actions': [],
            'rewards': [],
            'log_probs': [],
            'next_states': [],
            'dones': []
        }

        while step < total_steps:
            raw_state = self.env.get_state().screen_buffer
            state = preprocess(raw_state, resolution=self.resolution)

            action, action_log_prob = self.agent.get_action(state)
            if self.actions is not None:
                action_tensor = torch.Tensor(action)
                selected_action = self.actions[int(torch.argmax(action_tensor).item())]
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

            trajectories['states'].append(state)
            trajectories['actions'].append(action)
            trajectories['rewards'].append(reward)
            trajectories['log_probs'].append(action_log_prob)
            trajectories['next_states'].append(next_state)
            trajectories['dones'].append(float(done))

            self.wandb_logger.log({'Reward': reward})

            if done:
                ep_reward = self.env.get_total_reward()
                self.wandb_logger.log({'Episode Reward': ep_reward})
                episode_rewards.append(ep_reward)
                self.env.new_episode()

            step += 1

        # Преобразуем все в тензоры
        states = torch.FloatTensor(np.array(trajectories['states']))
        actions = torch.tensor(np.array(trajectories['actions']))
        rewards = torch.FloatTensor(np.array(trajectories['rewards']))
        next_states = torch.FloatTensor(np.array(trajectories['next_states']))
        dones = torch.FloatTensor(np.array(trajectories['dones']))
        log_probs = torch.cat(trajectories['log_probs'])

        dataset_size = states.size(0)
        indices = np.arange(dataset_size)

        for epoch in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, minibatch_size):
                end = start + minibatch_size
                mb_idx = indices[start:end]

                batch = {
                    'states': states[mb_idx],
                    'actions': actions[mb_idx],
                    'rewards': rewards[mb_idx],
                    'next_states': next_states[mb_idx],
                    'dones': dones[mb_idx],
                    'log_probs': log_probs[mb_idx]
                }

                policy_loss, value_loss, diagnostics = self.agent.train_agent(**batch)

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
