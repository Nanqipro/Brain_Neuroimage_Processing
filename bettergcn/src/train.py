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

def train_evaluate(model, train_loader, test_loader, optimizer, criterion, 
                  epochs=150, patience=20, class_weights=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    if class_weights is not None:
        class_weights = class_weights.to(device)
        weighted_criterion = torch.nn.NLLLoss(weight=class_weights)
    else:
        weighted_criterion = criterion
    
    # 余弦退火学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-5
    )
    
    # add early stopping
    best_test_acc = 0
    best_model = None
    no_improve = 0
    
    train_losses, train_accs, test_accs = [], [], []
    
    # 使用新版本的GradScaler API
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    
    for epoch in range(1, epochs + 1):
        # training
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            if scaler is not None:
                # 使用新版本的autocast API
                with torch.amp.autocast('cuda'):
                    out = model(data)
                    loss = weighted_criterion(out, data.y)
                
                scaler.scale(loss).backward()
                
                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                # regular training
                out = model(data)
                loss = weighted_criterion(out, data.y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            total_loss += loss.item() * data.num_graphs
            pred = out.argmax(dim=1)
            correct += pred.eq(data.y).sum().item()
            total += data.y.size(0)
        
        train_loss = total_loss / len(train_loader.dataset)
        train_acc = correct / total
        
        # evaluate
        test_acc, test_f1 = evaluate(model, test_loader, device)
        
        scheduler.step()
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'Epoch {epoch:03d}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Test Acc: {test_acc:.4f}, F1: {test_f1:.4f}, LR: {current_lr:.6f}')
        
        # best model selection
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model = model.state_dict().copy()
            no_improve = 0
        else:
            no_improve += 1
        
        # early stopping
        if no_improve >= patience:
            print(f"早停: {patience}个epoch没有改进")
            break
    
    if best_model is not None:
        model.load_state_dict(best_model)
        print(f"The accuracy of the best model: {best_test_acc:.4f}")
    
    return train_losses, train_accs, test_accs

def evaluate(model, loader, device):
    from sklearn.metrics import f1_score

    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            pred = out.argmax(dim=1)
            correct += pred.eq(data.y).sum().item()
            total += data.y.size(0)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
    
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return correct / total, f1

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

def plot_results(train_losses, train_accs, test_accs, save_path=None):
    """绘制训练结果"""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 4))
    
    # loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # 绘制准确率
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()