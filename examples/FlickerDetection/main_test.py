from FlickerDetection.get_dataloaders import get_dataloaders
from FlickerDetection.models.resnet_nonlocal import get_model
from FlickerDetection.test import test
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Start testing on: ',device)
_, _, test_loader = get_dataloaders()
model = get_model('./checkpoints/r3d101_K_200ep.pth', True, './checkpoints/epoch=29-val_loss=0.11.ckpt')
test(test_loader, model)