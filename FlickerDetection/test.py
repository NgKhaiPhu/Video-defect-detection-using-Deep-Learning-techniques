import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(test_loader, model):
    model.eval()
    model.to(device)
    total = len(test_loader.dataset)
    with torch.no_grad():
        correct = 0
        for X_test, y_test in test_loader:
            X_test = X_test.to(device)
            y_test = y_test.to(device)
            pred = model(X_test)
            y_test = y_test if pred.size() == y_test.size() else y_test.unsqueeze(-1)
            predicted = torch.round(pred)
            correct += (predicted == y_test).sum()
    print(f'Test accuracy: {correct.item()} / {total} = {correct.item()*100 / total:7.3f}%')