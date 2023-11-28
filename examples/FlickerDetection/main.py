from repo.get_dataloaders import get_dataloaders
from repo.models.resnet_nonlocal import get_model
from repo.train_configs import get_train_configs
from repo.train import train
from repo.test import test
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Start training on: ',device)
epochs = 1
train_loader, val_loader, test_loader = get_dataloaders()
model = get_model('./model/r3d101_K_200ep.pth',pretrained=False)
criterion, optimizer, scheduler = get_train_configs(model.parameters())

# remove last argument if you do not want to save checkpoint
# train(epochs, train_loader, val_loader, model, criterion, optimizer, scheduler)

train(epochs, train_loader, val_loader, model, criterion, optimizer, scheduler, './test_save/')
print("Training complete")

test(test_loader, model)