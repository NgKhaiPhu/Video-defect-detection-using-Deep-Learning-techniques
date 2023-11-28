from torchvision import transforms
from torchvision.transforms import (
    Compose,
    Lambda,
    Resize,
)
import pytorchvideo
from pytorchvideo.transforms import (
    Normalize,
)
import numpy as np
import torch

def frame_to_tensor(frames):
    return torch.tensor(np.array(frames)).to(torch.float32).permute(3, 0, 1, 2)
    
def inference_preprocess(frames):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    resize_to = (224, 224)
    
    transform = Compose([Lambda(lambda x: x/255.0), Normalize(mean, std), Resize(resize_to,antialias=True)])
    return transform(frame_to_tensor(frames)).unsqueeze(0)