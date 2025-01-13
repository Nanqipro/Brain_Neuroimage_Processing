import pandas as pd
import numpy as np
import os

def assign_labels(frame_timestamps, frame_files, excel_path, output_csv):
    """
    根据.xlsx文件中的时间戳和行为标签，为每一帧图像打上标签。

    参数：
    - frame_timestamps: list of float, 每帧对应的时间戳列表（秒）。
    - frame_files: list of str, 帧文件名列表。
    - excel_path: str, Excel文件路径，包含'stamp'和'Behavior'两列。
    - output_csv: str, 输出的标签CSV文件路径。

    返回：
    - None
    """
    # 读取Excel文件
    print("读取Excel文件...")
    df = pd.read_excel(excel_path)
    # 假设Excel文件中有'stamp'和'Behavior'两列
    # 确保'stamp'为浮点数，'Behavior'为整数或类别
    df = df[['stamp', 'Behavior']].dropna().sort_values('stamp').reset_index(drop=True)
    
    # 初始化标签列表
    labels = [0] * len(frame_files)  # 假设默认标签为0
    
    # 遍历每一帧，判断是否有行为发生在该帧的时间窗口内
    # 定义每帧的时间窗口为 [timestamp - window/2, timestamp + window/2)
    frame_interval = 1 / 4.8  # 约0.208秒
    half_window = frame_interval / 2  # 约0.104秒
    
    # 将行为时间戳和标签转为列表
    behavior_times = df['stamp'].tolist()
    behavior_labels = df['Behavior'].tolist()
    
    print("开始为每一帧打标签...")
    behavior_idx = 0  # 行为事件索引
    num_behaviors = len(behavior_times)
    
    for idx, (frame_time, frame_file) in enumerate(zip(frame_timestamps, frame_files)):
        window_start = frame_time - half_window
        window_end = frame_time + half_window
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

# 示例调用
# excel_path = 'path_to_your_labels.xlsx'
# output_labels_csv = 'path_to_save_labels.csv'
# assign_labels(frame_timestamps, frame_files, excel_path, output_labels_csv)
