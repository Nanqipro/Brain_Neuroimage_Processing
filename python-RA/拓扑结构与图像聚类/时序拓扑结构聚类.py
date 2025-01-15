import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# 设置Matplotlib中文字体
from matplotlib.font_manager import FontProperties
from matplotlib import rcParams

# 设置中文字体路径（根据你的操作系统调整）
# Windows 路径
font_path = 'C:/Windows/Fonts/simhei.ttf'
# Linux 路径
# font_path = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
# macOS 路径
# font_path = '/System/Library/Fonts/STHeiti Medium.ttc'

# 配置中文字体
font_prop = FontProperties(fname=font_path)
rcParams['font.family'] = font_prop.get_name()
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 步骤1：读取Excel数据
file_path = r'C:\Users\PAN\PycharmProjects\GitHub\python-RA\拓扑结构与图像聚类\topology_matrix.xlsx'  # 替换为你的文件路径
df = pd.read_excel(file_path)

# 查看数据结构
print("原始数据：")
print(df.head())

# 步骤2：数据预处理
timestamp_col = 'Time_Stamp'
feature_cols = [col for col in df.columns if col != timestamp_col]

X = df[feature_cols].values
timestamps = df[timestamp_col].values

# 检查是否有缺失值
if np.isnan(X).any():
    print("数据中存在缺失值，正在填充缺失值为0")
    X = np.nan_to_num(X)

# 步骤3：特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 步骤4：选择聚类数目
def determine_optimal_k(X, max_k=10):
    inertia = []
    silhouette = []
    K_range = range(2, max_k + 1)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
        score = silhouette_score(X, kmeans.labels_)
        silhouette.append(score)
    # 绘制肘部法和轮廓系数
    fig, ax1 = plt.subplots(figsize=(10, 5))
    color = 'tab:blue'
    ax1.set_xlabel('聚类数目 K')
    ax1.set_ylabel('惯性(Inertia)', color=color)
    ax1.plot(K_range, inertia, 'o-', color=color, label='惯性')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('轮廓系数(Silhouette Score)', color=color)
    ax2.plot(K_range, silhouette, 's--', color=color, label='轮廓系数')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    plt.title('确定最佳聚类数目 K')
    plt.show()

    return inertia, silhouette


# 确定最佳K值
determine_optimal_k(X_scaled, max_k=10)

# 根据图形选择K值，例如选择K=2
optimal_k = 6

# 步骤5：进行聚类
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(X_scaled)
labels = kmeans.labels_

# 将聚类标签添加到原始DataFrame
df['Cluster'] = labels
print("\n带有聚类标签的数据：")
print(df.head())

# 步骤6：可视化聚类结果
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 使用PCA降维到2维
pca = PCA(n_components=3, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# 或者使用t-SNE降维
# tsne = TSNE(n_components=2, random_state=42)
# X_tsne = tsne.fit_transform(X_scaled)

# 将降维后的数据添加到DataFrame
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

# 绘制散点图
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='Set1', s=100, alpha=0.7)

# 优化标签显示，减少密集
# 例如，仅标注每隔N个点，或者使用带有偏移的标签
label_step = max(1, len(df) // 20)  # 每隔20个点标注一个
for i, txt in enumerate(df[timestamp_col]):
    if i % label_step == 0:
        plt.text(df['PCA1'][i] + 0.02, df['PCA2'][i] + 0.02, str(txt), fontsize=9)

plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('聚类结果可视化 (PCA)')
plt.legend(title='Cluster')
plt.tight_layout()
plt.show()

# 如果需要完全交互式的标签，可以考虑使用Plotly
import plotly.express as px

fig = px.scatter(df, x='PCA1', y='PCA2', color='Cluster', text=timestamp_col,
                 title='聚类结果可视化 (PCA)',
                 hover_data=[timestamp_col] + feature_cols)

# 仅显示部分标签，以防过于密集
fig.update_traces(textposition='top center', textfont_size=10)
# 如果标签过于密集，可以不显示文本，只在hover时显示
# fig = px.scatter(df, x='PCA1', y='PCA2', color='Cluster',
#                  title='聚类结果可视化 (PCA)',
#                  hover_data=[timestamp_col] + feature_cols)
fig.show()

# 步骤7：分析聚类结果
for cluster in range(optimal_k):
    cluster_timestamps = df[df['Cluster'] == cluster][timestamp_col].values
    print(f"\nCluster {cluster} 包含的时间戳：")
    print(cluster_timestamps)
