from VideoNoiseDetection.train import train
from VideoNoiseDetection.get_impulse_noise_dataloaders import get_dataloaders
from VideoNoiseDetection.models.impulse_noise import get_model
from VideoNoiseDetection.train_configs import get_train_configs

epochs = 1

# get data loaders
train_loader, val_loader = get_dataloaders('./VideoNoiseDetection/datasets/SIDD_Small_sRGB_Only/Data/')

# get model
model = get_model()

# get all training configs
criterion, optimizer, scheduler = get_train_configs(model.parameters())

train(epochs, train_loader, val_loader, model, criterion, optimizer, scheduler)
print("Training complete")