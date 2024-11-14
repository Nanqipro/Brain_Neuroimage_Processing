import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # 用于显示进度条

# 视频路径和输出路径
video_path = r'C:\Users\PAN\Desktop\RA\2024年9月30日-脑神经图像处理\homecage.avi'  # 替换为实际视频路径
output_path = 'neuron_positions.png'  # 保存神经元位置图的路径

# 钙离子浓度阈值，用于检测闪光
FLASH_THRESHOLD = 150  # 可根据实际情况调整

# DBSCAN 聚类参数
CLUSTER_EPS = 5  # DBSCAN 的邻域距离
MIN_SAMPLES = 5  # DBSCAN 最小聚类点数

# 打开视频文件
cap = cv2.VideoCapture(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# 提取视频中的所有帧
frames = []
for _ in tqdm(range(frame_count), desc="Loading video frames"):
    ret, frame = cap.read()
    if not ret:
        break
    # 将帧转换为灰度图并存储
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frames.append(gray_frame)
cap.release()

# 检测每一帧中的“闪光”位置
def detect_flash_positions(frame):
    # 获取当前帧中大于阈值的像素位置
    return np.column_stack(np.where(frame > FLASH_THRESHOLD))

# 使用多线程加速“闪光”检测
all_positions = []
with ThreadPoolExecutor() as executor:
    futures = {executor.submit(detect_flash_positions, frame): i for i, frame in enumerate(frames)}
    for future in tqdm(as_completed(futures), total=len(futures), desc="Detecting flash positions"):
        flash_positions = future.result()
        all_positions.extend(flash_positions)

# 转换为 NumPy 数组
all_positions = np.array(all_positions)

# 使用 DBSCAN 聚类来识别不同的神经元位置
db = DBSCAN(eps=CLUSTER_EPS, min_samples=MIN_SAMPLES).fit(all_positions)
labels = db.labels_

# 获取唯一神经元位置
unique_positions = []
for label in set(labels):
    if label == -1:
        continue  # 跳过噪声点
    neuron_position = all_positions[labels == label].mean(axis=0)
    unique_positions.append(neuron_position)

unique_positions = np.array(unique_positions)

# 绘制结果
plt.figure(figsize=(8, 8))
plt.imshow(np.zeros((frame_height, frame_width)), cmap='gray')  # 模拟背景
plt.scatter(unique_positions[:, 1], unique_positions[:, 0], c='r', s=50, label='Neuron Positions')
for i, pos in enumerate(unique_positions):
    plt.text(pos[1], pos[0], f'N{i+1}', color='yellow', fontsize=8)  # 神经元编号

plt.title('Detected Neuron Positions')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.savefig(output_path)

print(f"神经元位置图已保存到: {output_path}")
