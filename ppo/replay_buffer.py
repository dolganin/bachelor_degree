import os
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity: int = 10000, momentum: float = 0.995, state_size: tuple = (3, 120, 130), action_dim: int = 20):
        """
        Initialize the replay buffer with decay.

        Args:
            capacity (int): Maximum capacity of the buffer.
            momentum (float): Decay factor for existing transitions.
            state_size (tuple): Shape of the state and next_state arrays.
            action_dim (int): Dimension of the action vector.
        """
        self.capacity = capacity
        self.momentum = momentum
        self.state_size = state_size
        self.action_dim = action_dim
        self.dump_path = "dump_buffer.npz"

        # Initialize arrays for each field with appropriate shapes and dtypes
        self.states = np.zeros((capacity, *state_size), dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_size), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.action_log_probs = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)

        self.pointer = 0
        self.size = 0

    def push(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray, reward: float, action_log_prob: float, done: bool):
        """
        Add a new transition to the buffer with decay applied to existing transitions.

        Args:
            state (np.ndarray): Current state.
            action (np.ndarray): Action vector.
            next_state (np.ndarray): Next state.
            reward (float): Reward for the action.
            combined_reward (float): Combined reward.
            action_log_prob (float): Log probability of the action.
            done (bool): Episode termination flag.
        """
        # Ensure action is a NumPy array with the correct shape

        # Apply decay to numerical fields excluding actions
        self.states *= self.momentum
        self.next_states *= self.momentum
        self.rewards *= self.momentum
        self.action_log_probs *= self.momentum

        # Add new transition
        self.states[self.pointer] = state
        self.actions[self.pointer] = action
        self.next_states[self.pointer] = next_state
        self.rewards[self.pointer] = reward
        self.action_log_probs[self.pointer] = action_log_prob
        self.dones[self.pointer] = done

        # Update pointer and size
        self.pointer = (self.pointer + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def sample(self, batch_size: int):
        """
        Sample a batch of transitions from the buffer.

        Args:
            batch_size (int): Size of the batch.

        Returns:
            Tuple[torch.Tensor, ...]: Batch of transitions as PyTorch tensors.
        """
        if self.size < batch_size:
            raise ValueError("Not enough transitions to sample the requested batch size.")

        # Randomly select indices
        indices = np.random.choice(self.size, batch_size, replace=False)

        # Gather transitions
        states = torch.FloatTensor(self.states[indices])
        actions = torch.FloatTensor(self.actions[indices])
        next_states = torch.FloatTensor(self.next_states[indices])
        rewards = torch.FloatTensor(self.rewards[indices])
        action_log_probs = torch.FloatTensor(self.action_log_probs[indices])
        dones = torch.BoolTensor(self.dones[indices])

        return states, actions, next_states, rewards, action_log_probs, dones

    def dump(self):
        """
        Save the buffer to disk and free up memory.
        """
        # Save all arrays to a single .npz file
        np.savez(self.dump_path,
                 states=self.states,
                 next_states=self.next_states,
                 actions=self.actions,
                 rewards=self.rewards,
                 action_log_probs=self.action_log_probs,
                 dones=self.dones,
                 pointer=self.pointer,
                 size=self.size)

        # Delete arrays to free up memory
        del self.states, self.next_states, self.actions, self.rewards, self.action_log_probs, self.dones
        self.states = None
        self.next_states = None
        self.actions = None
        self.rewards = None
        self.action_log_probs = None
        self.dones = None
        self.pointer = 0
        self.size = 0
        print("Buffer has been dumped to disk and memory has been freed.")

    def load(self):
        """
        Load the buffer from disk.
        """
        if not os.path.exists(self.dump_path):
            print("No buffer file found to load. Buffer remains empty.")
            return

        # Load all arrays from the .npz file
        data = np.load(self.dump_path)
        self.states = data['states']
        self.next_states = data['next_states']
        self.actions = data['actions']
        self.rewards = data['rewards']
        self.action_log_probs = data['action_log_probs']
        self.dones = data['dones']
        self.pointer = data['pointer']
        self.size = data['size']
        print("Buffer has been loaded from disk.")

    def __len__(self):
        return self.size