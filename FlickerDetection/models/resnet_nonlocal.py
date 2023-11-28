import torch
import torch.nn as nn
import torch.nn.functional as F
from repo.models import resnet
from repo.models.non_local import NLBlockND

class R3D_Attn(nn.Module):
    def __init__(self, pretrained_resnet):
        super().__init__()
        backbone = resnet.generate_model(101,n_classes=700)
        backbone.load_state_dict(pretrained_resnet)
        layers = list(backbone.children())[:-2]
        self.feature_extractor = nn.Sequential(*layers)
        # 16, 2048, 1, 7, 7
        self.nl = NLBlockND(in_channels=2048, mode='dot', dimension=3, bn_layer=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(100352, 64),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64,1)
        )

    def forward(self,x):
        self.feature_extractor.eval()
        with torch.no_grad():
            x = self.feature_extractor(x)
        x = self.nl(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return F.sigmoid(x)

def get_model(resnet3d_pretrained_path, pretrained=False, model_pretrained_path=None):
    """
    Specify ResNet3D-101 checkpoint path at resnet3d_pretrained_path.
    If pretrained=True, specify model checkpoint path at model_pretrained_path.
    """
    resnet3d_checkpoint = torch.load(resnet3d_pretrained_path)
    model = R3D_Attn(resnet3d_checkpoint['state_dict'])
    if pretrained:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_checkpoint = torch.load(model_pretrained_path,map_location=device)
        model.load_state_dict(model_checkpoint['state_dict'])
        model.to(device)

    return model
    