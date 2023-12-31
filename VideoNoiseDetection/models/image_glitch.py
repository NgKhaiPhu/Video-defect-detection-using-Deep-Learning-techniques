import torch
import torch.nn as nn
import torch.nn.functional as F

def get_model():
    class GlitchDetect(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3,16,3,1)
            self.conv2 = nn.Conv2d(16,32,3,1)
            self.conv3 = nn.Conv2d(32,64,3,1)
            self.conv4 = nn.Conv2d(64,128,3,1)
            self.conv5 = nn.Conv2d(128,256,3,1)
            self.fc1 = nn.Linear(6400,2056)
            self.dropout1 = nn.Dropout(0.2)
            self.fc2 = nn.Linear(2056,512)
            self.dropout2 = nn.Dropout(0.2)
            self.fc3 = nn.Linear(512,1)  
            
        def forward(self,x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x,2,2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x,2,2)
            x = F.relu(self.conv3(x))
            x = F.max_pool2d(x,2,2)
            x = F.relu(self.conv4(x))
            x = F.max_pool2d(x,2,2)
            x = F.relu(self.conv5(x))
            x = F.max_pool2d(x,2,2)
    
            x = x.view(-1,6400)
    
            x = F.relu(self.fc1(x))
            x = self.dropout1(x)
            x = F.relu(self.fc2(x))
            x = self.dropout2(x)
            x = self.fc3(x)
            return F.sigmoid(x)
    
    model = GlitchDetect()
    return model