import torch
import numpy as np
from tqdm import trange
from torch.nn import Module
from base.trainer_base import TrainerRL
from utilities.preprocessing import preprocess
from server_consumer.broker_kafka import publish_data
from utilities.video_logger import VideoLogger
from base.agent_evaluator import AgentEvaluator
from collections import deque

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
        # Для скользящего окна средней награды за 100 эпизодов
        recent_episode_rewards = deque(maxlen=100)

        loss_dict = {}
        obs = self.env.reset()
        total_reward = 0.0
        step = 0

        # Буфер rollout-а
        trajectories = {
            'states': [], 'actions': [], 'rewards': [],
            'log_probs': [], 'next_states': [], 'dones': []
        }

        # Собираем rollout
        while step < total_steps:
            # предобработка состояний
            batch_states = [preprocess(o, resolution=self.resolution) for o in obs]
            # получаем батч действий и их лог-вероятности
            out = [self.agent.get_action(s) for s in batch_states]
            actions, logps = zip(*out)

            # приводим индексы к списку команд
            selected = []
            for a in actions:
                if self.actions:
                    idx = int(torch.argmax(torch.Tensor(a)).item())
                    selected.append(self.actions[idx])
                else:
                    selected.append(a)

            # делаем шаг во всех средах
            next_obs, rewards, dones, infos = self.env.step(selected)

            # сохраняем переходы и считаем эпизодические награды
            for i in range(self.n_envs):
                trajectories['states'].append(batch_states[i])
                trajectories['actions'].append(actions[i])
                trajectories['rewards'].append(rewards[i])
                trajectories['log_probs'].append(logps[i])

                if not dones[i]:
                    ns = preprocess(next_obs[i], resolution=self.resolution)
                else:
                    ns = np.zeros((3, *self.resolution), dtype=np.float32)
                trajectories['next_states'].append(ns)
                trajectories['dones'].append(float(dones[i]))

                total_reward += rewards[i]

                # если эпизод закончился — логируем его награду
                if dones[i]:
                    ep_reward = infos[i].get('episode_reward', None)
                    # если в info нет, можно считать сумму последних trajectories['rewards'] или делать reset в env
                    recent_episode_rewards.append(ep_reward)
                    # логируем скользящую среднюю
                    if len(recent_episode_rewards) == 100:
                        avg100 = sum(recent_episode_rewards) / 100.0
                        self.wandb_logger.log({'AvgRewardLast100': avg100})

            obs = next_obs
            step += self.n_envs

        # конвертация rollout-буфера в тензоры
        S  = torch.FloatTensor(np.array(trajectories['states']))
        A  = torch.tensor(np.array(trajectories['actions']))
        R  = torch.FloatTensor(np.array(trajectories['rewards']))
        NS = torch.FloatTensor(np.array(trajectories['next_states']))
        D  = torch.FloatTensor(np.array(trajectories['dones']))
        LP = torch.cat(trajectories['log_probs'])

        # PPO-эпохи
        idxs = np.arange(S.size(0))
        for _ in range(self.ppo_epochs):
            np.random.shuffle(idxs)
            for start in range(0, len(idxs), batch_size):
                mb = idxs[start:start+batch_size]
                batch = {
                    'states': S[mb], 'actions': A[mb], 'rewards': R[mb],
                    'next_states': NS[mb], 'dones': D[mb], 'log_probs': LP[mb]
                }
                p_loss, v_loss, diag = self.agent.train_agent(**batch)
                loss_dict.setdefault('policy_loss', []).append(p_loss)
                loss_dict.setdefault('value_loss', []).append(v_loss)
                self.wandb_logger.log({
                    'Train policy loss': p_loss,
                    'Train value loss':   v_loss,
                    'Policy Entropy':     diag.get('entropy', 0.0),
                    'Advantages Mean':    diag.get('advantages_mean', 0.0)
                })

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
