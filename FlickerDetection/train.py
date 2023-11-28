import torch
from .val import validate
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(epochs, train_loader, val_loader, model, criterion, optimizer, scheduler, checkpoint_path=None):
    """
    If checkpoint_path is not specified, train() will not save checkpoint.
    If checkpoint_path is specified, train() will first try to create the folder by the specified path, before running the train loop.
    """
    save_ckpt = False
    if checkpoint_path != None:
        try:
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
        except OSError:
            print ('Error: Creating directory of data')
            return
        save_ckpt = True
    
    model.to(device)
    for i in range(epochs):
        model.train()
        for b, (X_train, y_train) in enumerate(train_loader):
            b+=1
            X_train = X_train.to(device)
            y_train = y_train.to(device)
            pred = model(X_train)
            loss = criterion(pred, y_train if pred.size() == y_train.size() else y_train.unsqueeze(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            if b % 10 == 0:
                print(f'epoch: {i:2}  batch: {b:4}  loss: {loss.item():10.8f}')

        if save_ckpt:
            torch.save({
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, os.path.join(checkpoint_path,f"epoch_{i}.ckpt"))

        loss = validate(val_loader, model, criterion)
        scheduler.step()
        