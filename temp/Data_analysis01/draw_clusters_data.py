import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the provided Excel files
trace_file_path = './data/smoothed_normalized_2979_CSDS_Day6.xlsx'
cluster_file_path = './data/kmeans_clustering_results_2979_CSDS_Day6.xlsx'

# Read the trace data
trace_df = pd.read_excel(trace_file_path)

# Read the clustering result data
clustered_df = pd.read_excel(cluster_file_path)

# 控制波动幅度的缩放因子和垂直间距
default_scaling_factor = 5  # 默认簇的缩放因子
increased_scaling_factor = 20  # 增大缩放因子，用于特殊cluster
vertical_separation = 20  # 垂直偏移的间距

# 获取每个独立的簇
unique_clusters = clustered_df['Cluster'].unique()

# 为每个簇生成一个图
for cluster_id in unique_clusters:
    # print(clustered_df.columns)
    # 获取当前簇中对应的神经元
    neuron_names = clustered_df[clustered_df['Cluster'] == cluster_id]['Neuron'].values

    # 为当前簇设置图表
    plt.figure(figsize=(10, 8))

    # 根据 cluster_id 选择适当的缩放因子
    scaling_factor = increased_scaling_factor if cluster_id == 1 else default_scaling_factor

    # 对神经元按编号顺序进行垂直偏移并绘制
    for idx, neuron in enumerate(sorted(neuron_names, key=lambda x: int(x[1:]))):
        # 绘制当前神经元的轨迹，应用缩放因子和垂直偏移
        plt.plot(trace_df['stamp'], trace_df[neuron] * scaling_factor + idx * vertical_separation, label=neuron)

    # 设置x轴范围
    plt.xlim(0, 3000)

    # # 添加垂直虚线
    # for x_pos in [100, 200, 300, 400]:
    #     plt.axvline(x=x_pos, color='red', linestyle='--', linewidth=0.8)  # 红色虚线
    #     plt.axvline(x=x_pos - 50, color='blue', linestyle='--', linewidth=0.8)  # 蓝色虚线

    # # 添加文本注释
    # plt.text(100, len(neuron_names) * vertical_separation, 'enclosed arms', ha='center', fontsize=12)
    # plt.text(400, len(neuron_names) * vertical_separation, 'open arms', ha='center', fontsize=12)

    # 设置标签和标题
    plt.xlabel('Stamp')
    plt.ylabel('Traces (ordered by neuron number)')
    plt.title(f'Traces of Neurons in Cluster {cluster_id}')

    # 显示图例
    plt.legend(loc='upper right', fontsize='small', ncol=2)

    # 调整布局并显示图表
    plt.tight_layout()
    plt.show()
