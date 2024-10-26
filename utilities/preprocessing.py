import torchvision.transforms as transforms
from torch import tensor

def preprocess(img: list, resolution: tuple) -> tensor:
    """Down samples image to resolution"""

    transformer = transforms.Compose(
        [   transforms.ToTensor(),
            transforms.Resize(tuple(resolution))
            ])
    img = transformer(img)
    return img