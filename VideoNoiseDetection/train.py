import torch
import torch.nn as nn
import torch.nn.functional as F
from .val import validate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(epochs, train_loader, val_loader, model, criterion, optimizer, scheduler):
    model.to(device)
    for i in range(epochs):
        for b, (X_train, y_train) in enumerate(train_loader):
            b+=1
            #push data to GPU
            X_train = X_train.to(device)
            y_train = y_train.to(device)
            # Apply the model
            pred = model(X_train)
            loss = criterion(pred, y_train)
            
            # Update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            # Print interim results
            if b % 10 == 0:
                print(f'epoch: {i:2}  batch: {b:4}  loss: {loss.item():10.8f}')

        loss = validate(val_loader, model, criterion)
        scheduler.step(loss)

        