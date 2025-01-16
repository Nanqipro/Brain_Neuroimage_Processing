import os
import tifffile
import pandas as pd
import numpy as np
from PIL import Image

# 添加以下代码来抑制警告
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 设置日志级别
tf.get_logger().setLevel('ERROR')  # 只显示错误信息

from tensorflow.keras.utils import to_categorical

def extract_frames(tif_path, output_folder, original_fps, desired_fps):
    """
    从.tif文件中按指定频率提取帧并保存。

    参数：
    - tif_path: str, 输入.tif文件路径。
    - output_folder: str, 保存提取帧的文件夹路径。
    - original_fps: float, 原始视频的帧率（例如24.0Hz）。
    - desired_fps: float, 期望提取的帧率（例如4.8Hz）。

    返回：
    - frame_files: list of str, 保存的帧文件名列表。
    - frame_timestamps: list of float, 每帧对应的时间戳列表（秒）。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 计算采样间隔
    sampling_interval = original_fps / desired_fps
    if sampling_interval < 1:
        print("期望帧率高于原始帧率，无法进行采样。")
        sampling_interval = 1  # 不采样，使用所有帧
    
    # 读取.tif文件
    print("读取.tif文件...")
    tif = tifffile.TiffFile(tif_path)
    num_frames = len(tif.pages)
    print(f".tif文件总帧数: {num_frames}")
    
    frame_files = []
    frame_timestamps = []
    for i, page in enumerate(tif.pages):
        if i % int(sampling_interval) == 0:
            # 提取帧
            img = page.asarray()
            # 转换为PIL图像
            img_pil = Image.fromarray(img)
            # 定义帧文件名
            frame_filename = f"frame_{i:05d}.png"
            frame_path = os.path.join(output_folder, frame_filename)
            img_pil.save(frame_path)
            frame_files.append(frame_filename)
            # 计算时间戳
            timestamp = i / original_fps
            frame_timestamps.append(timestamp)
            print(f"保存 {frame_filename} 于 {timestamp:.3f} 秒")
    tif.close()
    print("帧提取完成。")
    return frame_files, frame_timestamps

def assign_labels(frame_timestamps, frame_files, excel_path, output_csv, desired_fps):
    """
    根据.xlsx文件中的时间戳和行为标签，为每一帧图像打上标签。

    参数：
    - frame_timestamps: list of float, 每帧对应的时间戳列表（秒）。
    - frame_files: list of str, 帧文件名列表。
    - excel_path: str, Excel文件路径，包含'stamp'和'Behavior'两列。
    - output_csv: str, 输出的标签CSV文件路径。
    - desired_fps: float, 期望的帧率。

    返回：
    - None
    """
    # 读取Excel文件
    print("读取Excel文件...")
    df = pd.read_excel(excel_path,sheet_name='CHB')
    # 确保Excel文件中有'stamp'和'Behavior'两列
    if not {'stamp', 'Behavior'}.issubset(df.columns):
        raise ValueError("Excel文件中必须包含'stamp'和'Behavior'两列。")
    
    # 清洗数据：删除缺失值并排序
    df = df[['stamp', 'Behavior']].dropna().sort_values('stamp').reset_index(drop=True)
    
    # 初始化标签列表
    labels = [0] * len(frame_files)  # 默认标签为0
    
    # 提取行为时间戳和标签
    behavior_times = df['stamp'].tolist()
    behavior_labels = df['Behavior'].tolist()
    
    print("开始为每一帧打标签...")
    behavior_idx = 0  # 行为事件索引
    num_behaviors = len(behavior_times)
    
    for idx, (frame_time, frame_file) in enumerate(zip(frame_timestamps, frame_files)):
        window_start = frame_time - (1 / desired_fps) / 2
        window_end = frame_time + (1 / desired_fps) / 2
        frame_label = 0  # 默认标签
        
        # 检查是否有行为事件在该时间窗口内
        while behavior_idx < num_behaviors and behavior_times[behavior_idx] < window_end:
            if behavior_times[behavior_idx] >= window_start:
                frame_label = behavior_labels[behavior_idx]
                break  # 假设每帧最多一个行为事件
            behavior_idx += 1
        
        labels[idx] = frame_label
        print(f"帧 {frame_file} 于 {frame_time:.3f} 秒被标记为 {frame_label}")
    
    # 创建标签DataFrame
    label_df = pd.DataFrame({
        'frame': frame_files,
        'timestamp': frame_timestamps,
        'label': labels
    })
    
    # 保存为CSV
    label_df.to_csv(output_csv, index=False)
    print(f"帧标签已保存到 {output_csv}")

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
        # 取窗口末尾的标签作为该体积的标签
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

def main():
    # 需要修改的路径和参数：
    
    # 1. .tif 视频文件路径
    tif_path = '../datasets/data/day6.tif'  # 修改为您的.tif文件实际路径
    
    # 2. 帧图像保存文件夹
    output_frames_folder = '../datasets/output/frames'  # 修改为您想保存提取帧的文件夹路径
    
    # 3. 视频帧率参数
    original_fps = 4.8  # 修改为您的视频实际帧率
    desired_fps = 4.8  # 修改为您想要的输出帧率
    
    # 4. 行为标签Excel文件路径
    excel_path = '../datasets/data/day6行为标记.xlsx'  # 修改为您的标签Excel文件实际路径
    
    # 5. 输出的标签CSV文件路径
    output_labels_csv = '../datasets/output/labels6.csv'  # 修改为您想保存标签CSV的路径
    
    # 其他参数（根据需要调整）：
    window_size = 10  # 3D数据窗口大小
    step = 5         # 滑动窗口步长
    num_classes = 2  # 行为类别数量
    
    # 步骤1：提取帧
    frame_files, frame_timestamps = extract_frames(tif_path, output_frames_folder, original_fps, desired_fps)
    
    # 步骤2：为帧打标签
    assign_labels(frame_timestamps, frame_files, excel_path, output_labels_csv, desired_fps)
    
    # 步骤3：创建3D数据集
    X, y = create_3d_dataset(output_frames_folder, output_labels_csv, window_size, step, num_classes)
    
    # 保存数据集
    np.save('../datasets/dataset/X_3d.npy', X)
    np.save('../datasets/dataset/y_labels.npy', y)
    print("3D数据集已保存为X_3d.npy和y_labels.npy")

if __name__ == "__main__":
    main()
