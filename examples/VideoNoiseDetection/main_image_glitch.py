from VideoNoiseDetection.train import train
from VideoNoiseDetection.get_image_glitch_dataloaders import get_dataloaders
from VideoNoiseDetection.models.image_glitch import get_model
from VideoNoiseDetection.train_configs import get_train_configs

epochs = 1

# get data loaders
train_loader, val_loader = get_dataloaders('./VideoNoiseDetection/datasets/505_video_frames/')

# get model
model = get_model()

# get all training configs
criterion, optimizer, scheduler = get_train_configs(model.parameters())

train(epochs, train_loader, val_loader, model, criterion, optimizer, scheduler)
print("Training complete")