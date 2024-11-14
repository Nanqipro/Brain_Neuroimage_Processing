import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from tqdm import tqdm

# 视频路径和输出路径
video_path = r'C:\Users\PAN\Desktop\RA\2024年9月30日-脑神经图像处理\homecage.avi'  # 替换为实际视频路径
output_path = 'neuron_positions_clustered.png'  # 保存神经元位置图的路径

# 钙离子浓度阈值，用于检测闪光30
FLASH_THRESHOLD = 180  # 根据实际情况调整

# DBSCAN 聚类参数
CLUSTER_EPS = 1  # 邻域距离
MIN_SAMPLES = 5  # 最小聚类点数

# 打开视频文件
cap = cv2.VideoCapture(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# 存储每帧的聚类中心
all_centers = []

# 逐帧处理并聚类
for _ in tqdm(range(0, frame_count, 1), desc="Processing video frames"):  # 每隔5帧处理一次
    ret, frame = cap.read()
    if not ret:
        break
    # 转换为灰度图像
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测高于阈值的区域
    high_intensity_points = np.column_stack(np.where(gray_frame > FLASH_THRESHOLD))

    if len(high_intensity_points) > 0:
        # 在每帧上聚类，提取中心点
        db = DBSCAN(eps=CLUSTER_EPS, min_samples=MIN_SAMPLES).fit(high_intensity_points)
        labels = db.labels_
        unique_labels = set(labels) - {-1}  # 忽略噪声点
        frame_centers = [high_intensity_points[labels == label].mean(axis=0) for label in unique_labels]
        all_centers.extend(frame_centers)

cap.release()

# 将逐帧聚类中心进一步聚类
all_centers = np.array(all_centers)
print(f"Total frame-based centers detected: {len(all_centers)}")
if len(all_centers) > 10000:
    all_centers = all_centers[::10]  # 降采样以加速处理

# 在所有中心点上再次聚类
final_db = DBSCAN(eps=15, min_samples=5).fit(all_centers)
final_labels = final_db.labels_

# 绘制最终聚类结果
plt.figure(figsize=(9, 9))
plt.imshow(np.zeros((frame_height, frame_width)), cmap='gray')  # 模拟背景
for label in set(final_labels):
    if label == -1:
        continue  # 跳过噪声点
    cluster_points = all_centers[final_labels == label]
    plt.scatter(cluster_points[:, 1], cluster_points[:, 0], s=20)

plt.title('Detected Neuron Positions Using Clustering')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.savefig(output_path)
plt.show()

print(f"Neuron position map saved to: {output_path}")








