import torch
from .val import validate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(epochs, train_loader, val_loader, model, criterion, optimizer, scheduler):
    model.to(device)
    for i in range(epochs):
        for b, (X_train, y_train) in enumerate(train_loader):
            b += 1
            X_train = X_train.to(device)
            y_train = y_train.to(device)
            pred = model(X_train)
            loss = criterion(pred, y_train if pred.size() == y_train.size() else y_train.unsqueeze(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            if b % 10 == 0:
                print(f'epoch: {i:2}  batch: {b:4}  loss: {loss.item():10.8f}')

        loss = validate(val_loader, model, criterion)
        scheduler.step(loss)

        