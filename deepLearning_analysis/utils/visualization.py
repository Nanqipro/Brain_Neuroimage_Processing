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