import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def validate(val_loader, model, criterion):
    running_loss = 0
    with torch.no_grad():
        for vb, (X_test, y_test) in enumerate(val_loader):
            #push data to GPU
            X_test = X_test.to(device)
            y_test = y_test.to(device)
            # Apply the model
            val = model(X_test)
            loss = criterion(val, y_test)
            running_loss += loss
    
    avg_loss = running_loss / (vb+1)
    print(f'Validation finished, loss: {avg_loss}')
    return avg_loss