import torch
import torch.nn as nn

def get_train_configs(params):    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(params,lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.1,patience=2,verbose=True)

    return criterion, optimizer, scheduler