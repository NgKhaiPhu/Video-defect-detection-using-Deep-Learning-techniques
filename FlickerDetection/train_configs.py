import torch
import torch.nn as nn

def get_train_configs(params):    
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(params, lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=50,verbose=True)

    return criterion, optimizer, scheduler