import torch.nn as nn
import torch

class PolicyNetwork(nn.Module):
    def __init__(self, shared_transformer: nn.Module, action_dim: int, hidden_dim: int = 64):
        super(PolicyNetwork, self).__init__()
        self.shared_transformer = shared_transformer
        self.fc_mean = nn.Sequential(
            nn.Linear(shared_transformer.embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.fc_log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, state: torch.Tensor):
        features = self.shared_transformer(state)  # (batch_size, embedding_dim)
        mean = self.fc_mean(features)  # (batch_size, action_dim)
        log_std = self.fc_log_std.expand_as(mean)
        std = torch.exp(log_std)
        return mean, std
