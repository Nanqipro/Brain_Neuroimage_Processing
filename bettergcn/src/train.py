import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch_geometric.loader import DataLoader



def train_model(model, train_loader, optimizer, device, class_weights=None):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    # 遍历所有批次
    for data in train_loader:
        try:
            # 这样写更安全，避免Data.to()方法的潜在问题
            data = data.to(device)  
        except TypeError:
            # 如果上面的方法失败，手动将每个张量移到设备上
            for key in data.keys:
                if torch.is_tensor(data[key]):
                    data[key] = data[key].to(device)
        # Forward pass
        # - x: 节点特征 (batch_size * 43, 1)
        # - edge_index: 边索引 (2, batch_size * avg_num_edges)
        # - batch: 节点所属批次 (batch_size * 43)
        # 输出维度: (batch_size, num_classes)
        optimizer.zero_grad()
        out = model(data)

        if class_weights is not None:
            weights = class_weights.to(device)
            loss = F.nll_loss(out, data.y, weight=weights)
        else:
            loss = F.nll_loss(out, data.y)

        # Backward pass
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        
        # 为每个批次收集预测和标签，而不仅仅是最后一个批次
        _, pred = out.max(dim=1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())

    # 计算整个训练集的各项指标，而不是只基于最后一个批次
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

def evaluate_model(model, loader, device):
    model.eval()
    correct = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in loader:
            try:
                data = data.to(device)
            except TypeError:
                for key in data.keys:
                    if torch.is_tensor(data[key]):
                        data[key] = data[key].to(device)
                        
            outputs = model(data)
            _, pred = outputs.max(dim=1)
            correct += pred.eq(data.y).sum().item()
            
            # 收集每个批次的预测结果和标签
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

def plot_confusion_matrix(y_true, y_pred, class_names, result_dir='result'):
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
    plt.savefig(f'{result_dir}/confusion_matrix.png')
    plt.close()

def plot_training_metrics(train_metrics, val_metrics, result_dir='result'):
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
    plt.savefig(f'{result_dir}/training_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_learning_curve(train_metrics, val_metrics, result_dir='result'):
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
    plt.savefig(f'{result_dir}/learning_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
