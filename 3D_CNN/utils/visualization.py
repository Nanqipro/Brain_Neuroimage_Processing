import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from pathlib import Path
import datetime
from config.config import Config
import numpy as np
import matplotlib as mpl

# 设置中文字体
def setup_chinese_font():
    try:
        # 尝试设置微软雅黑
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        try:
            # 尝试设置思源黑体
            plt.rcParams['font.sans-serif'] = ['Source Han Sans CN']
            plt.rcParams['axes.unicode_minus'] = False
        except:
            try:
                # 尝试设置文泉驿微米黑
                plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
                plt.rcParams['axes.unicode_minus'] = False
            except:
                # 如果都失败了，使用英文标签
                use_english_labels = True
                return use_english_labels
    return False

# 调用字体设置函数
use_english_labels = setup_chinese_font()

def plot_training_history(
    train_losses: List[float],
    val_losses: List[float],
    train_accuracies: List[float],
    val_accuracies: List[float]
) -> None:
    """绘制训练历史并保存图片"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    plt.figure(figsize=(12, 5))
    
    # 绘制损失
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='train loss')
    plt.plot(val_losses, label='val loss')
    plt.title('train and val loss')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend()
    
    # 绘制准确率
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='train acc')
    plt.plot(val_accuracies, label='val acc')
    plt.title('train and val acc')
    plt.xlabel('Epoch')
    plt.ylabel('acc')
    plt.legend()
    
    # 保存图片
    save_path = Config.VISUALIZATION_DIR / f'training_history_{timestamp}.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def visualize_sample(
    frames_folder: Path,
    labels_csv: Path,
    sample_idx: int = 0,
    window_size: int = 10,
    step: int = 5
) -> None:
    """可视化样本数据并保存图片"""
    import pandas as pd
    from PIL import Image
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    label_df = pd.read_csv(labels_csv)
    frame_files = label_df['frame'].tolist()
    labels = label_df['label'].tolist()

    start_idx = sample_idx * step
    end_idx = start_idx + window_size
    sample_frames = frame_files[start_idx:end_idx]
    sample_label = labels[end_idx - 1]

    fig, axes = plt.subplots(1, window_size, figsize=(20, 2))
    for i, frame in enumerate(sample_frames):
        img = Image.open(frames_folder / frame).convert('L')
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
    plt.suptitle(f"Sample {sample_idx} Label: {sample_label}" if use_english_labels else f"样本 {sample_idx} 标签: {sample_label}")
    
    # 保存图片
    save_path = Config.VISUALIZATION_DIR / f'sample_visualization_{timestamp}.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    classes: List[str],
    normalize: bool = False
) -> None:
    """绘制混淆矩阵并保存图片"""
    from sklearn.metrics import confusion_matrix
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes
    )
    plt.title('normalized confusion matrix' if normalize else 'confusion matrix')
    plt.ylabel('true label')
    plt.xlabel('predicted label')
    
    # 保存图片
    save_path = Config.VISUALIZATION_DIR / f'confusion_matrix_{timestamp}.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def visualize_feature_maps(model, layer_name, sample_idx=0, test_loader=None, device='cpu'):
    """
    可视化特征图并保存。
    
    参数：
    - model: PyTorch模型。
    - layer_name: str, 需要可视化的层名称。
    - sample_idx: int, 测试样本索引。
    - test_loader: DataLoader, 测试数据加载器。
    - device: str, 运行模型的设备。
    
    返回：
    - None
    """
    import torch
    activation = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    # 注册钩子
    for name, layer in model.named_modules():
        if name == layer_name:
            layer.register_forward_hook(get_activation(name))
    
    # 获取样本数据
    model.eval()
    with torch.no_grad():
        sample_input, _ = test_loader.dataset[sample_idx]
        sample_input = sample_input.unsqueeze(0).to(device)  # 添加批量维度
        output = model(sample_input)
    
    # 获取特征图
    if layer_name in activation:
        feature_maps = activation[layer_name].cpu().numpy()  # (1, C, T, H, W)
        print(f"特征图形状: {feature_maps.shape}")
    else:
        print(f"未找到层名称: {layer_name}")
        return
    
    # 可视化特征图
    time_step = feature_maps.shape[3] // 2  # 中间时间步
    num_features = min(8, feature_maps.shape[1])  # 可视化前8个特征图
    
    plt.figure(figsize=(20, 5))
    for i in range(num_features):
        plt.subplot(1, num_features, i+1)
        plt.imshow(feature_maps[0, i, time_step, :, :], cmap='viridis')
        plt.axis('off')
        plt.title(f'Filter {i+1}')
    plt.suptitle(f"特征映射 - 层 {layer_name}, 时间步 {time_step}")
    
    # 保存图片
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = Config.VISUALIZATION_DIR / f'feature_maps_{layer_name}_sample{sample_idx}_{timestamp}.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"特征图已保存到: {save_path}")

def generate_gradcam(model, layer_name, sample_idx=0, class_idx=None, test_loader=None, device='cpu'):
    """
    Generate Grad-CAM heatmap.
    
    参数：
    - model: PyTorch模型。
    - layer_name: str, 需要可视化的层名称。
    - sample_idx: int, 测试样本索引。
    - class_idx: int, 预测类别索引。默认为None，使用模型的预测类别。
    - test_loader: DataLoader, 测试数据的加载器。
    - device: str, 运行模型的设备。
    
    返回：
    - heatmap: numpy.ndarray, 热力图数组，形状为 (T', H', W')。
    """
    import torch
    activation = {}
    gradients = {}
    
    def forward_hook(module, input, output):
        activation[layer_name] = output

    def backward_hook(module, grad_in, grad_out):
        gradients[layer_name] = grad_out[0]
    
    # 注册钩子
    for name, layer in model.named_modules():
        if name == layer_name:
            layer.register_forward_hook(forward_hook)
            layer.register_backward_hook(backward_hook)
    
    # 获取样本数据
    model.eval()
    sample_input, label = test_loader.dataset[sample_idx]
    sample_input = sample_input.unsqueeze(0).to(device)
    sample_input.requires_grad = True
    
    # 前向传播
    output = model(sample_input)
    
    if class_idx is None:
        class_idx = torch.argmax(output, dim=1).item()
    
    # 计算损失
    loss = output[0, class_idx]
    
    # 反向传播
    model.zero_grad()
    loss.backward()
    
    # 获取梯度和激活
    if layer_name in gradients and layer_name in activation:
        grads = gradients[layer_name].detach().cpu().numpy()[0]  # 添加 detach()
        act = activation[layer_name].detach().cpu().numpy()[0]   # 添加 detach()
    else:
        print(f"Layer name not found: {layer_name}")
        return None
    
    # 计算权重
    weights = np.mean(grads, axis=(1, 2, 3))  # (C,)
    
    # 生成热力图
    heatmap = np.zeros(act.shape[1:])  # (T, H, W)
    for i, w in enumerate(weights):
        heatmap += w * act[i, :, :, :]
    
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1
    
    return heatmap


def display_gradcam(heatmap, slice_idx=None, cmap='viridis'):
    """
    显示Grad-CAM热力图。
    
    参数：
    - heatmap: numpy.ndarray, 热力图数组，形状为 (T', H', W')。
    - slice_idx: int, 指定时间步索引。默认为None，选择中间切片。
    - cmap: str, 颜色映射方案。
    
    返回：
    - None
    """
    if heatmap is None:
        print("No heatmap to display.")
        return

    if slice_idx is None:
        slice_idx = heatmap.shape[0] // 2  # 中间时间步
    
    plt.figure(figsize=(6,6))
    plt.imshow(heatmap[slice_idx], cmap=cmap)
    plt.title(f"Grad-CAM Heatmap - Time Step {slice_idx}")
    plt.colorbar()
    plt.show()


def visualize_gradcam(model, layer_name, sample_idx=0, test_loader=None, device='cpu'):
    """
    可视化并保存Grad-CAM热力图。
    
    参数：
    - model: PyTorch模型对象。
    - layer_name: str, 需要可视化的卷积层名称。
    - sample_idx: int, 测试样本索引。
    - test_loader: DataLoader, 测试数据的加载器。
    - device: str, 运行模型的设备。
    
    返回：
    - None
    """
    heatmap = generate_gradcam(model, layer_name, sample_idx=sample_idx, test_loader=test_loader, device=device)
    if heatmap is not None:
        slice_idx = heatmap.shape[0] // 2
        plt.figure(figsize=(6,6))
        plt.imshow(heatmap[slice_idx], cmap='viridis')
        plt.title(f"Grad-CAM Heatmap - Time Step {slice_idx}")
        plt.colorbar()
        
        # 保存图片
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = Config.VISUALIZATION_DIR / f'gradcam_{layer_name}_sample{sample_idx}_{timestamp}.png'
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Grad-CAM heatmap saved to: {save_path}")
    else:
        print("No heatmap to display.") 