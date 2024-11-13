import pandas as pd
import numpy as np
from scipy.spatial.distance import directed_hausdorff
import plotly.express as px
from sklearn.cluster import KMeans
from numba import jit, cuda
from tqdm import tqdm
import math

# 查看系统中的 GPU 设备
devices = cuda.list_devices()
print("可用的 GPU 设备:")
for i, device in enumerate(devices):
    print(f"设备 {i}: {device.name}")

# 选择独立显卡（假设其设备编号为 1，如果不确定，请查看输出确认独立显卡的编号）
device_id = 0  # 独立显卡的设备编号（可能需要调整）
cuda.select_device(device_id)
print(f"已选择 GPU 设备: {cuda.get_current_device().name}")

# 读取滑动窗口数据（假设数据已包含每个滑动窗口的特征）
file_path = 'calcium_window_emd_cluster_single_sheet.xlsx'
sheet_name = 'Window_30_Step_5'  # 指定要读取的工作表名称
df = pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")

# 提取特征（假设特征列为 'Mean', 'StdDev', 'Peak'）
data = df[['Mean', 'StdDev', 'Peak']].values

# 初始化聚类参数
n_clusters = 5
max_iter = 100

# 随机初始化质心
np.random.seed(42)
initial_centers = data[np.random.choice(data.shape[0], n_clusters, replace=False)]


# 定义 GPU 上运行的 Hausdorff 距离计算函数
@cuda.jit
def hausdorff_distance_kernel(A, B, max_distances):
    i, j = cuda.grid(2)  # 获取二维线程索引
    if i < A.shape[0] and j < B.shape[0]:
        dist = 0
        for k in range(A.shape[1]):
            dist += (A[i, k] - B[j, k]) ** 2
        dist = math.sqrt(dist)

        # 使用原子操作更新 max_distances
        cuda.atomic.max(max_distances, i, dist)


def hausdorff_gpu(A, B):
    A_device = cuda.to_device(A)
    B_device = cuda.to_device(B)
    max_distances = cuda.device_array(A.shape[0], dtype=np.float32)
    max_distances[:] = 0  # 初始化为 0

    # 设置每块线程数和网格大小，确保充分利用 GPU 资源
    threads_per_block = (32, 32)
    blocks_per_grid_x = (36000 + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (36000 + threads_per_block[1] - 1) // threads_per_block[1]

    hausdorff_distance_kernel[(blocks_per_grid_x, blocks_per_grid_y), threads_per_block](A_device, B_device,
                                                                                         max_distances)

    return max_distances.copy_to_host()


# 聚类函数
def hausdorff_kmeans(data, centers, max_iter=100):
    labels = np.zeros(len(data), dtype=int)

    for iteration in tqdm(range(max_iter), desc="Clustering Progress"):
        # 分配样本到最接近的质心（基于 Hausdorff 距离）
        for i, point in enumerate(data):
            distances = [
                max(
                    hausdorff_gpu(np.array([point]), np.array([center]))[0],
                    hausdorff_gpu(np.array([center]), np.array([point]))[0]
                )
                for center in centers
            ]
            labels[i] = np.argmin(distances)

        # 更新质心
        new_centers = []
        for j in range(n_clusters):
            cluster_points = data[labels == j]
            if len(cluster_points) > 0:
                new_center = np.mean(cluster_points, axis=0)
                new_centers.append(new_center)
            else:
                new_centers.append(centers[j])

        # 检查收敛
        new_centers = np.array(new_centers)
        if np.allclose(new_centers, centers):
            break
        centers = new_centers

    return labels, centers

# 执行基于 Hausdorff 距离的 K-means 聚类
labels, centers = hausdorff_kmeans(data, initial_centers, max_iter=max_iter)

# 将聚类标签添加到原始 DataFrame
df['Cluster'] = labels

# 使用 Plotly 绘制 3D 散点图
fig = px.scatter_3d(
    df,
    x='Start Time',  # 使用时间窗口的起始时间作为 x 轴
    y='Neuron',  # 不同神经元作为 y 轴
    z='Cluster',  # 聚类类别作为 z 轴
    color='Cluster',  # 根据聚类标签着色
    title='Neuron Clustering across Time Windows with Hausdorff Distance',
    labels={'Start Time': 'Time Window Start', 'Neuron': 'Neuron', 'Cluster': 'Cluster'}
)

fig.update_layout(scene=dict(
    xaxis_title='Time Window Start',
    yaxis_title='Neuron',
    zaxis_title='Cluster'
))

fig.show()
