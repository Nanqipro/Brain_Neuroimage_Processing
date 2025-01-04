import numpy as np
import pandas as pd
import os
from tensorflow.keras.utils import to_categorical

def create_3d_dataset(frames_folder, labels_csv, window_size=10, step=5, num_classes=2):
    """
    创建3D CNN训练数据集。

    参数：
    - frames_folder: str, 分帧图像所在文件夹路径。
    - labels_csv: str, 带标签的CSV文件路径。
    - window_size: int, 每个3D体积包含的帧数。
    - step: int, 窗口滑动步长。
    - num_classes: int, 行为类别数量。

    返回：
    - X: numpy.ndarray, 3D CNN的输入数据，形状为 (样本数, T, H, W, C)。
    - y: numpy.ndarray, 标签，形状为 (样本数, num_classes)。
    """
    # 读取标签CSV
    print("读取标签CSV文件...")
    label_df = pd.read_csv(labels_csv)
    
    # 获取所有帧文件名和标签
    frame_files = label_df['frame'].tolist()
    labels = label_df['label'].tolist()
    
    # 获取帧的完整路径
    frame_paths = [os.path.join(frames_folder, f) for f in frame_files]
    
    # 获取图像尺寸和通道数
    sample_image = Image.open(frame_paths[0])
    H, W = sample_image.size
    C = 1  # 灰度图
    print(f"图像尺寸: {H}x{W}, 通道数: {C}")
    
    # 初始化数据列表
    X = []
    y = []
    
    # 滑动窗口提取3D体积
    print("开始构建3D数据集...")
    for i in range(0, len(frame_paths) - window_size + 1, step):
        window_frames = frame_paths[i:i+window_size]
        window_labels = labels[i:i+window_size]
        # 假设标签取窗口末尾的标签作为该体积的标签
        window_label = window_labels[-1]
        
        # 读取并堆叠帧图像
        volume = []
        for frame_path in window_frames:
            img = Image.open(frame_path).convert('L')  # 灰度图
            img_array = np.array(img, dtype=np.float32) / 255.0  # 归一化
            img_array = img_array.reshape(H, W, 1)  # 添加通道维度
            volume.append(img_array)
        volume = np.stack(volume, axis=0)  # T x H x W x C
        X.append(volume)
        y.append(window_label)
    
    X = np.array(X)
    y = np.array(y)
    print(f"构建完成，样本数: {X.shape[0]}")
    
    # 转换标签为分类形式
    if num_classes > 2:
        y = to_categorical(y, num_classes=num_classes)
    else:
        y = y.reshape(-1, 1)  # 二分类时保持形状
    
    return X, y

# 示例调用
# frames_folder = 'path_to_save_frames'
# labels_csv = 'path_to_save_labels.csv'
# window_size = 10
# step = 5
# num_classes = 2
# X, y = create_3d_dataset(frames_folder, labels_csv, window_size, step, num_classes)
