import torch.nn as nn
import torch

class SharedTransformer(nn.Module):
    def __init__(
        self,
        image_channels: int = 1,
        image_height: int = 120,  # Указываем высоту
        image_width: int = 130,   # Указываем ширину
        patch_size: int = 4,
        embedding_dim: int = 148,
        num_heads: int = 8,
        num_layers: int = 6,
        mlp_dim: int = 256,
        dropout: float = 0.1
    ) -> None:
        super(SharedTransformer, self).__init__()
        
        assert image_height % patch_size == 0 and image_width % patch_size == 0, "Image dimensions must be divisible by patch size."
        self.patch_size = patch_size

        # Количество патчей по высоте и ширине
        self.num_patches_height = image_height // patch_size
        self.num_patches_width = image_width // patch_size

        # Общее количество патчей
        self.num_patches = self.num_patches_height * self.num_patches_width
        self.embedding_dim = embedding_dim

        # Patch Embedding
        self.patch_embed = nn.Conv2d(
            in_channels=image_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # Positional Encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, self.num_patches, embedding_dim))
        nn.init.trunc_normal_(self.positional_encoding, std=0.02)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        # x имеет форму: (batch_size, image_channels, image_height, image_width)
        x = self.patch_embed(x)  # Shape: (batch_size, embedding_dim, num_patches_height, num_patches_width)
        x = x.flatten(2)          # Shape: (batch_size, embedding_dim, num_patches)
        x = x.transpose(1, 2)     # Shape: (batch_size, num_patches, embedding_dim)

        # Добавляем positional encoding
        x = x + self.positional_encoding 

        x = x.transpose(0, 1)  # Shape: (num_patches, batch_size, embedding_dim)

        # Pass through transformer
        x = self.transformer_encoder(x)  # Shape: (num_patches, batch_size, embedding_dim)

        x = x.mean(dim=0)  # Shape: (batch_size, embedding_dim)

        return x
