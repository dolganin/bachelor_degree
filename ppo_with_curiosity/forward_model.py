import torch
import torch.nn as nn
import torchvision.models as models

class ForwardModelCNN(nn.Module):
    """
    Forward Model с использованием ResNet34 в качестве энкодера и стандартного CNN-декодера.
    Предсказывает следующее состояние (изображение) на основе текущего состояния и действия.
    """
    
    def __init__(self, action_dim: int, image_channels: int = 1, image_size: int = 84, hidden_dim: int = 512):
        super(ForwardModelCNN, self).__init__()
        
        # Инициализация ResNet34 в качестве энкодера
        resnet = models.resnet34(pretrained=True)
        # Извлекаем только слои до последнего слоя классификации
        modules = list(resnet.children())[:-2]  # Убираем avgpool и fc
        self.encoder = nn.Sequential(*modules)
        self.image_size = image_size
        
        # Заморозка весов энкодера, если необходимо
        # for param in self.encoder.parameters():
        #     param.requires_grad = False
        
        # Полносвязный слой для объединения с действием
        self.action_fc = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Полносвязный слой для объединения признаков из энкодера и действия
        self.combined_fc = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Декодер: Восстановление изображения из объединённых признаков
        self.decoder = nn.Sequential(
            # Первый транспонированный свёрточный слой
            nn.ConvTranspose2d(hidden_dim, 256, kernel_size=4, stride=2, padding=1),  # (256, 4, 4)
            nn.ReLU(),
            nn.BatchNorm2d(256),
            
            # Второй транспонированный свёрточный слой
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # (128, 8, 8)
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            # Третий транспонированный свёрточный слой
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (64, 16, 16)
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            # Четвёртый транспонированный свёрточный слой
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # (32, 32, 32)
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            # Пятый транспонированный свёрточный слой
            nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2, padding=1),  # (image_channels, 64, 64)
            nn.Sigmoid()  # Нормализуем выходные значения в диапазон [0, 1]
        )
        
        # Дополнительный свёрточный слой для достижения нужного размера изображения
        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(image_channels, image_channels, kernel_size=4, stride=2, padding=1),  # (image_channels, 128, 128)
            nn.Sigmoid()
        )
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через модель.
        
        Args:
            state (torch.Tensor): Текущее состояние (изображение) размером (batch_size, C, H, W).
            action (torch.Tensor): Действие размером (batch_size, action_dim).
        
        Returns:
            torch.Tensor: Предсказанное следующее состояние (изображение) размером (batch_size, C, H, W).
        """
        # Пропуск текущего состояния через энкодер ResNet34
        encoded_state = self.encoder(state)  # Shape: (batch_size, 512, H', W')
        
        # Извлечение признаков из энкодера
        batch_size, enc_channels, enc_height, enc_width = encoded_state.size()
        encoded_state = encoded_state.view(batch_size, enc_channels, -1).mean(dim=2)  # (batch_size, 512)
        
        # Пропуск действия через полносвязный слой
        action_embedding = self.action_fc(action)  # (batch_size, 512)
        
        # Объединение признаков состояния и действия
        combined = torch.cat([encoded_state, action_embedding], dim=1)  # (batch_size, 1024)
        combined = self.combined_fc(combined)  # (batch_size, 512)
        
        # Преобразование объединённых признаков в тензор для декодера
        combined = combined.view(batch_size, -1, 1, 1)  # (batch_size, 512, 1, 1)
        
        # Пропуск через декодер
        decoded = self.decoder(combined)  # (batch_size, image_channels, 64, 64)
        
        # Дополнительное преобразование для достижения нужного размера
        predicted_next_state = self.final_conv(decoded)  # (batch_size, image_channels, 128, 128)
        
        # Если требуется размер 84x84, можно использовать интерполяцию или изменить архитектуру декодера
        # Например, использовать nn.Upsample или изменить параметры ConvTranspose2d слоёв
        
        # Пример интерполяции до нужного размера:
        predicted_next_state = nn.functional.interpolate(predicted_next_state, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        
        return predicted_next_state
