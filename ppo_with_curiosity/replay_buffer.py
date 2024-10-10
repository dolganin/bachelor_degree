from collections import deque
import random
import torch

class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, next_state):
        self.buffer.append((state, action, next_state))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, next_states = zip(*batch)
        return torch.FloatTensor(states), torch.FloatTensor(actions), torch.FloatTensor(next_states)
    
    def __len__(self):
        return len(self.buffer)
