import torch.nn as nn
from torch import Tensor

class ValueNetwork(nn.Module):
    def __init__(self, shared_transformer: nn.Module, hidden_dim: int = 64):
        super(ValueNetwork, self).__init__()
        self.shared_transformer = shared_transformer
        self.value_head = nn.Sequential(
            nn.Linear(shared_transformer.embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: Tensor):
        features = self.shared_transformer(state)  # (batch_size, embedding_dim)
        value = self.value_head(features)  # (batch_size, 1)
        return value