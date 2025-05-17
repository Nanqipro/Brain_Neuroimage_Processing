import torch
import matplotlib.pyplot as plt
from datetime import datetime
import os

def train_epoch(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)

def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data in loader:
            out = model(data)
            pred = out.argmax(dim=1)
            correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)

def train_evaluate(model, train_loader, test_loader, optimizer, criterion, epochs=50):
    train_losses, train_accs, test_accs = [], [], []
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion)
        train_losses.append(loss)
        train_acc = evaluate(model, train_loader)
        train_accs.append(train_acc)
        test_acc = evaluate(model, test_loader)
        test_accs.append(test_acc)
        print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

    return train_losses, train_accs, test_accs

def save_model(model, optimizer, epoch, train_acc, test_acc, save_dir='models'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{save_dir}/model_epoch{epoch}_{timestamp}_acc{test_acc:.4f}.pt"
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_acc': train_acc,
        'test_acc': test_acc,
    }, filename)
    print(f'Model saved to {filename}')
    return filename

def plot_results(train_losses, train_accs, test_accs, save_dir='results'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    # plot losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'r-', label='Train Acc')
    plt.plot(epochs, test_accs, 'g-', label='Test Acc')
    plt.title('Training and Test Accuracies')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{save_dir}/results_{timestamp}.png"
    plt.savefig(filename)
    plt.close()
    print(f'Plot saved to {filename}')