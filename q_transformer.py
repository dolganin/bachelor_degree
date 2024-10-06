import torch
import torch.nn as nn

class DuelQNet(nn.Module):
    """
    Dueling Deep Q-Network (Duel DQN) using a Vision Transformer (ViT) architecture.
    """

    def __init__(
        self,
        available_actions_count: int,
        image_channels: int = 1,
        image_size: int = 84,
        patch_size: int = 7,
        embedding_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 6,
        mlp_dim: int = 256,
        dropout: float = 0.1
    ) -> None:
        super(DuelQNet, self).__init__()
        
        assert image_size % patch_size == 0, "Image size must be divisible by patch size."
        self.patch_size = patch_size
        self.num_patches = 24
        self.embedding_dim = embedding_dim

        # Patch Embedding: Преобразуем изображение в патчи и создаем эмбеддинги для каждого патча
        self.patch_embed = nn.Conv2d(
            in_channels=image_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # Позиционное кодирование
        self.positional_encoding = nn.Parameter(torch.zeros(1, self.num_patches, embedding_dim))
        nn.init.trunc_normal_(self.positional_encoding, std=0.02)

        # Трансформер
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Линейные слои для расчёта State Value и Advantage
        self.state_fc = nn.Sequential(
            nn.Linear(embedding_dim, 64),  # Используем полный embedding_dim
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.advantage_fc = nn.Sequential(
            nn.Linear(embedding_dim, 64),  # Используем полный embedding_dim
            nn.ReLU(),
            nn.Linear(64, available_actions_count)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass через трансформер с раздельной обработкой для State Value и Advantage.
        """
        batch_size = x.size(0)

        x = self.patch_embed(x)  # Shape: (batch_size, embedding_dim, num_patches_sqrt, num_patches_sqrt)
        x = x.flatten(2)          # Shape: (batch_size, embedding_dim, num_patches)
        x = x.transpose(1, 2)     # Shape: (batch_size, num_patches, embedding_dim)

        x = x + self.positional_encoding 

        x = x.transpose(0, 1)  # Shape: (num_patches, batch_size, embedding_dim)

        # Пропускаем через трансформер
        x = self.transformer_encoder(x)  # Shape: (num_patches, batch_size, embedding_dim)

        x = x.mean(dim=0)  # Shape: (batch_size, embedding_dim)

        state_value = self.state_fc(x).reshape(-1, 1)  # Используем все признаки для state_value

        advantage_values = self.advantage_fc(x)  # Используем все признаки для advantage_values

        # Финальная сборка Q-значений
        q_values = state_value + (advantage_values - advantage_values.mean(dim=1, keepdim=True))

        return q_values
