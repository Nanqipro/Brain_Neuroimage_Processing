import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

def train_epoch(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        
        # 收集预测和标签用于计算指标
        _, pred = out.max(dim=1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())
    
    # 计算各项指标
    train_accuracy = accuracy_score(all_labels, all_preds)
    train_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    train_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    train_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    metrics = {
        'loss': total_loss / len(train_loader.dataset),
        'accuracy': train_accuracy,
        'precision': train_precision,
        'recall': train_recall,
        'f1': train_f1
    }
    
    return metrics

def evaluate(model, loader):
    model.eval()
    correct = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data in loader:
            out = model(data)
            pred = out.argmax(dim=1)
            correct += pred.eq(data.y).sum().item()
            
            # 收集预测和标签用于计算指标
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
    
    # 计算各项指标
    accuracy = correct / len(loader.dataset)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': all_preds,
        'labels': all_labels
    }
    
    return metrics

def train_evaluate(model, train_loader, test_loader, optimizer, criterion, epochs=50):
    # 初始化指标字典
    train_metrics = {
        'loss': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    val_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    # 存储最后一个epoch的预测结果用于混淆矩阵
    final_predictions = None
    final_labels = None
    
    for epoch in range(1, epochs + 1):
        # 训练阶段
        train_result = train_epoch(model, train_loader, optimizer, criterion)
        
        # 记录训练指标
        train_metrics['loss'].append(train_result['loss'])
        train_metrics['accuracy'].append(train_result['accuracy'])
        train_metrics['precision'].append(train_result['precision'])
        train_metrics['recall'].append(train_result['recall'])
        train_metrics['f1'].append(train_result['f1'])
        
        # 验证阶段
        val_result = evaluate(model, test_loader)
        
        # 记录验证指标
        val_metrics['accuracy'].append(val_result['accuracy'])
        val_metrics['precision'].append(val_result['precision'])
        val_metrics['recall'].append(val_result['recall'])
        val_metrics['f1'].append(val_result['f1'])
        
        # 保存最后一个epoch的预测结果
        if epoch == epochs:
            final_predictions = val_result['predictions']
            final_labels = val_result['labels']
        
        print(f'Epoch {epoch:03d}, Loss: {train_result["loss"]:.4f}, Train Acc: {train_result["accuracy"]:.4f}, Test Acc: {val_result["accuracy"]:.4f}')

    return train_metrics, val_metrics, final_predictions, final_labels

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

def setup_matplotlib_fonts():
    """设置matplotlib字体和样式"""
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Arial']  # 优先使用中文黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    # 设置坐标轴标签字体大小和粗细 - 进一步增大字体
    plt.rcParams['axes.labelsize'] = 24  # 坐标轴标签字体大小
    plt.rcParams['axes.labelweight'] = 'bold'  # 坐标轴标签字体粗细
    plt.rcParams['xtick.labelsize'] = 20  # X轴刻度字体大小
    plt.rcParams['ytick.labelsize'] = 20  # Y轴刻度字体大小
    plt.rcParams['axes.titlesize'] = 28  # 图表标题字体大小
    plt.rcParams['axes.titleweight'] = 'bold'  # 图表标题字体粗细
    plt.rcParams['legend.fontsize'] = 20  # 图例字体大小
    
    # 设置线条样式
    plt.rcParams['lines.linewidth'] = 3  # 默认线条粗细

def plot_confusion_matrix(y_true, y_pred, class_names, result_dir='results'):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)

    # 确保class_names和混淆矩阵大小匹配
    if len(class_names) != cm.shape[0]:
        print(f"警告：类别名称列表({len(class_names)})与混淆矩阵大小({cm.shape[0]})不匹配")
        # 如果不匹配，使用默认类别标签
        class_names = [f"类别{i}" for i in range(cm.shape[0])]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label', fontsize=24, fontweight='bold')
    plt.ylabel('True Label', fontsize=24, fontweight='bold') 
    plt.title('Confusion Matrix', fontsize=28, fontweight='bold')
    plt.xticks(fontsize=20, fontweight='bold')
    plt.yticks(fontsize=20, fontweight='bold')
    plt.tight_layout()
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    plt.savefig(f'{result_dir}/confusion_matrix.png')
    plt.close()

def plot_training_metrics(train_metrics, val_metrics, result_dir='results'):
    """绘制训练过程中的各项指标"""
    metrics = ['loss', 'accuracy', 'precision', 'recall', 'f1']
    fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 5*len(metrics)))
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        epochs = range(len(train_metrics[metric]) if metric in train_metrics else len(val_metrics[metric]))
        
        if metric in train_metrics:
            # 绘制粗线条
            line1 = ax.plot(epochs, train_metrics[metric], 'b-', linewidth=3, 
                           label=f'Train {metric}', alpha=0.8, zorder=2)
            # 每10个epoch添加红点
            marker_epochs = [e for e in epochs if e % 10 == 0 or e == 0]
            marker_values = [train_metrics[metric][e] for e in marker_epochs]
            ax.scatter(marker_epochs, marker_values, color='red', s=80, 
                      zorder=3, edgecolors='darkred', linewidth=2)
        
        if metric in val_metrics:
            # 绘制粗线条
            line2 = ax.plot(epochs, val_metrics[metric], 'g-', linewidth=3, 
                           label=f'Validation {metric}', alpha=0.8, zorder=2)
            # 每10个epoch添加红点
            marker_epochs = [e for e in epochs if e % 10 == 0 or e == 0]
            marker_values = [val_metrics[metric][e] for e in marker_epochs]
            ax.scatter(marker_epochs, marker_values, color='red', s=80, 
                      zorder=3, edgecolors='darkred', linewidth=2)
            
        # 增强立体感 - 添加阴影效果
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#f8f9fa')
        
        # 调整字体大小
        ax.set_title(f'{metric.capitalize()} over epochs', fontsize=26, fontweight='bold', pad=20)
        ax.set_xlabel('Epochs', fontsize=24, fontweight='bold')
        ax.set_ylabel(metric.capitalize(), fontsize=24, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=20, width=2, length=6)
        ax.legend(fontsize=20, frameon=True, fancybox=True, shadow=True, loc='upper right', bbox_to_anchor=(1.0, 0.95))
        
        # 添加边框样式
        for spine in ax.spines.values():
            spine.set_linewidth(2)
    
    plt.tight_layout(pad=3.0)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    plt.savefig(f'{result_dir}/training_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_learning_curve(train_metrics, val_metrics, result_dir='results'):
    """绘制学习曲线（训练损失和验证准确率）"""
    plt.figure(figsize=(14, 8))
    
    # 创建两个Y轴
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    epochs = range(len(train_metrics['loss']))
    
    # 绘制训练损失 - 粗线条
    line1 = ax1.plot(epochs, train_metrics['loss'], 'b-', linewidth=4, 
                     label='Train Loss', alpha=0.8, zorder=2)
    # 每10个epoch添加红点
    marker_epochs = [e for e in epochs if e % 10 == 0 or e == 0]
    marker_loss_values = [train_metrics['loss'][e] for e in marker_epochs]
    ax1.scatter(marker_epochs, marker_loss_values, color='red', s=100, 
               zorder=3, edgecolors='darkred', linewidth=2)
    
    ax1.set_xlabel('Epochs', fontsize=24, fontweight='bold')
    ax1.set_ylabel('Loss', color='b', fontsize=24, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='b', labelsize=20, width=2, length=6)
    ax1.tick_params(axis='x', labelsize=20, width=2, length=6)
    
    # 绘制验证准确率 - 粗线条
    line2 = ax2.plot(epochs, val_metrics['accuracy'], 'g-', linewidth=4, 
                     label='Validation Accuracy', alpha=0.8, zorder=2)
    # 每10个epoch添加红点
    marker_acc_values = [val_metrics['accuracy'][e] for e in marker_epochs]
    ax2.scatter(marker_epochs, marker_acc_values, color='red', s=100, 
               zorder=3, edgecolors='darkred', linewidth=2)
    
    ax2.set_ylabel('Accuracy', color='g', fontsize=24, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='g', labelsize=20, width=2, length=6)
    
    # 增强立体感
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_facecolor('#f8f9fa')
    
    # 添加边框样式
    for spine in ax1.spines.values():
        spine.set_linewidth(2)
    for spine in ax2.spines.values():
        spine.set_linewidth(2)
    
    # 添加图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=20, 
                 frameon=True, fancybox=True, shadow=True, bbox_to_anchor=(1.0, 0.95))
    
    plt.title('Training Loss and Validation Accuracy', fontsize=28, fontweight='bold', pad=20)
    plt.tight_layout(pad=3.0)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    plt.savefig(f'{result_dir}/learning_curve.png', dpi=300, bbox_inches='tight')
    plt.close()