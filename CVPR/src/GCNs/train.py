import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from torch_geometric.data import DataLoader



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
        # - x: 节点特征 (batch_size * n, 1)
        # - edge_index: 边索引 (2, batch_size * avg_num_edges)
        # - batch: 节点所属批次 (batch_size * n)
        # 输出维度: (batch_size, num_classes)
        optimizer.zero_grad()
        out = model(data)

        # 处理 OGB 数据集的标签格式 (batch_size, 1) -> (batch_size,)
        y = data.y.squeeze() if data.y.dim() > 1 else data.y

        if class_weights is not None:
            weights = class_weights.to(device)
            loss = F.nll_loss(out, y, weight=weights)
        else:
            loss = F.nll_loss(out, y)

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
    all_probs = []  # 存储预测概率用于计算 AUC-ROC

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
            
            # 计算预测概率（用于 AUC-ROC）
            probs = torch.exp(outputs)  # 从 log_softmax 转换为概率
            
            # 处理 OGB 数据集的标签格式 (batch_size, 1) -> (batch_size,)
            y = data.y.squeeze() if data.y.dim() > 1 else data.y
            
            correct += pred.eq(y).sum().item()
            
            # 收集每个批次的预测结果和标签
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # 计算各项指标
    accuracy = correct / len(loader.dataset)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # 计算 AUC-ROC
    auc_roc = 0.0
    try:
        all_probs_array = np.array(all_probs)
        all_labels_array = np.array(all_labels)
        
        # 检查类别数
        num_classes = all_probs_array.shape[1] if len(all_probs_array.shape) > 1 else 2
        unique_labels = np.unique(all_labels_array)
        
        if len(unique_labels) > 1:  # 至少有两个类别才能计算 AUC-ROC
            if num_classes == 2:
                # 二分类：使用正类的概率
                auc_roc = roc_auc_score(all_labels_array, all_probs_array[:, 1])
            else:
                # 多分类：使用 ovr (one-vs-rest) 策略
                auc_roc = roc_auc_score(all_labels_array, all_probs_array, multi_class='ovr', average='weighted')
        else:
            auc_roc = 0.0  # 只有一个类别，无法计算 AUC-ROC
    except Exception as e:
        # 如果计算失败，设置为 0
        auc_roc = 0.0
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, class_names, result_dir='result'):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'{result_dir}/confusion_matrix.png')
    plt.close()

def plot_training_metrics(train_metrics, val_metrics, result_dir='result'):
    """绘制训练过程中的各项指标"""
    metrics = ['loss', 'accuracy', 'precision', 'recall', 'f1']
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4*len(metrics)))
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        if metric in train_metrics:
            ax.plot(train_metrics[metric], 'b-', label=f'Train {metric}')
        
        if metric in val_metrics:
            ax.plot(val_metrics[metric], 'r-', label=f'Validation {metric}')
            
        ax.set_title(f'{metric.capitalize()} over epochs')
        ax.set_xlabel('Epochs')
        ax.set_ylabel(metric.capitalize())
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{result_dir}/training_metrics.png')
    plt.close()

# def plot_learning_curve(train_metrics, val_metrics, result_dir='result'):
#     """绘制学习曲线（训练损失和验证准确率）"""
#     plt.figure(figsize=(12, 6))
    
#     # 创建两个Y轴
#     ax1 = plt.gca()
#     ax2 = ax1.twinx()
    
#     # 绘制训练损失
#     ax1.plot(train_metrics['loss'], 'b-', label='Train Loss')
#     ax1.set_xlabel('Epochs')
#     ax1.set_ylabel('Loss', color='b')
#     ax1.tick_params(axis='y', labelcolor='b')
    
#     # 绘制验证准确率
#     ax2.plot(val_metrics['accuracy'], 'r-', label='Validation Accuracy')
#     ax2.set_ylabel('Accuracy', color='r')
#     ax2.tick_params(axis='y', labelcolor='r')
    
#     # 添加图例
#     lines1, labels1 = ax1.get_legend_handles_labels()
#     lines2, labels2 = ax2.get_legend_handles_labels()
#     ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
#     plt.title('Training Loss and Validation Accuracy')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(f'{result_dir}/learning_curve.png')
#     plt.close()
