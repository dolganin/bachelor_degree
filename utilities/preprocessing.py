import torch
import torch.nn.functional as F
import numpy as np

def preprocess(img: np.ndarray, resolution: tuple) -> torch.Tensor:
    """
    Преобразует np.ndarray изображения (H, W, C) или (C, H, W)
    в тензор (C, H_new, W_new) с нужным разрешением.
    """
    if img.ndim == 3 and img.shape[0] in [1, 3]:  # (C, H, W)
        img_tensor = torch.from_numpy(img).float()
    elif img.ndim == 3 and img.shape[2] in [1, 3]:  # (H, W, C)
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    # Добавляем фейковый батч
    img_tensor = img_tensor.unsqueeze(0)  # (1, C, H, W)
    img_tensor = F.interpolate(img_tensor, size=resolution, mode='bilinear', align_corners=False)
    return img_tensor.squeeze(0)  # (C, H_new, W_new)
