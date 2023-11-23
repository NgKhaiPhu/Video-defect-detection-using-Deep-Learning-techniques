from VideoNoiseDetection.train import train
from VideoNoiseDetection.get_impulse_noise_dataloader import get_dataloaders
from VideoNoiseDetection.models.impulse_noise import get_model
from VideoNoiseDetection.train_configs import get_train_configs

epochs = 1
train_loader, val_loader = get_dataloaders('./VideoNoiseDetection/datasets/SIDD_Small_sRGB_Only/Data/')
model = get_model()
criterion, optimizer, scheduler = get_train_configs(model.parameters())
train(epochs, train_loader, val_loader, model, criterion, optimizer, scheduler)
print("Training complete")