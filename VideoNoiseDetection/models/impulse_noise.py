import torch
import torch.nn as nn
import torch.nn.functional as F

def get_model():
    class AutoEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            
            self.encoder = nn.Sequential(
                nn.Conv2d(3,32,3,1,padding='same'),
                nn.ReLU(),
                nn.MaxPool2d(2,2),
                nn.Conv2d(32,64,3,1,padding='same'),
                nn.ReLU(),
                nn.MaxPool2d(2,2),
                nn.Conv2d(64,128,3,1,padding='same'),
                nn.ReLU(),
                nn.MaxPool2d(2,2)
            )
            
            self.decoder = torch.nn.Sequential(
                nn.ConvTranspose2d(128,128,3,2,padding=1,output_padding=1),
                nn.ConvTranspose2d(128,64,3,2,padding=1,output_padding=1),
                nn.ConvTranspose2d(64,32,3,2,padding=1,output_padding=1),
                nn.Conv2d(32,3,3,1,padding='same'),
                nn.Sigmoid()
            )
     
        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded
    
    model = AutoEncoder()
    return model
    
    