import os
import tifffile
import pandas as pd
import numpy as np
from PIL import Image

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

# 示例调用
# tif_path = 'path_to_your_video.tif'
# output_frames_folder = 'path_to_save_frames'
# original_fps = 24.0  # 请根据实际情况修改
# desired_fps = 4.8
# frame_files, frame_timestamps = extract_frames(tif_path, output_frames_folder, original_fps, desired_fps)
