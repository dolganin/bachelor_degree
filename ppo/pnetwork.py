import torch.nn as nn
import torch

class PolicyNetwork(nn.Module):
    def __init__(self, shared_transformer: nn.Module, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.shared_transformer = shared_transformer
        self.fc = nn.Sequential(
            nn.Linear(shared_transformer.embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    def forward(self, state: torch.Tensor):
        features = self.shared_transformer(state)  # (batch_size, embedding_dim)
        logits = self.fc(features)  # (batch_size, action_dim), без активации
        return logits

