import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def validate(val_loader, model, criterion):
    running_loss = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for vb, (X_val, y_val) in enumerate(val_loader):
            X_val = X_val.to(device)
            y_val = y_val.to(device)
            pred = model(X_val)
            y_val = y_val if pred.size() == y_val.size() else y_val.unsqueeze(-1)
            predicted = torch.round(pred)
            correct += (predicted == y_val).sum()
            loss = criterion(pred, y_val)
            running_loss += loss

    avg_loss = running_loss / (vb+1)
    acc = correct.item()*100 / len(val_loader.dataset)
    print(f'Validation finished, loss: {avg_loss.item()} / {X_val.size()[0]}')
    print(f'Test accuracy: {correct.item()}/{len(val_loader.dataset)} = {acc:7.3f}%')
    return avg_loss