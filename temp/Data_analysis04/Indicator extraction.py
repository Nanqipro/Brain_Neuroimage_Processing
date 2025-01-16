# # 按滑动窗口顺序存储
# import pandas as pd
# import numpy as np
#
# # 加载数据
# file_path = './data/Day3.xlsx'  # 请替换为实际文件路径
# data = pd.read_excel(file_path)
#
# # 确保只处理数值列并忽略 FrameLost 列
# neuron_columns = data.columns[1:]  # 假设 n1~n53 为神经元数据
# data[neuron_columns] = data[neuron_columns].apply(pd.to_numeric, errors='coerce')
#
# # 计算阈值
# threshold = data[neuron_columns].mean().mean() + data[neuron_columns].std().mean()
#
# # 定义滑动窗口参数
# window_size = 30
# step_size = 5
#
#
# # 定义计算关键指标的函数
# def calculate_metrics_for_neurons(data, threshold, window_size, step_size):
#     results = []
#     neuron_columns = data.columns[1:53]  # 只针对神经元列
#     time_stamps = data['stamp']
#
#     for start in range(0, len(data) - window_size + 1, step_size):
#         window_end = start + window_size
#         window_data = data.iloc[start:window_end]
#
#         start_time = time_stamps[start]  # 获取窗口的开始时间戳
#
#         for neuron in neuron_columns:
#             neuron_signal = window_data[neuron]
#
#             # 1. Amplitude: 窗口内信号的最大值
#             amplitude = neuron_signal.max()
#
#             # 2. Duration (above threshold)
#             duration = (neuron_signal > threshold).sum()
#
#             # 3. Event frequency: 比例
#             frequency = duration / len(neuron_signal)
#
#             # 4. Rise time 和 5. Decay time 的计算
#             rising_times = []
#             decaying_times = []
#             above_threshold = neuron_signal > threshold
#             transitions = np.diff(above_threshold.astype(int))
#             rising_indices = np.where(transitions == 1)[0]
#             decaying_indices = np.where(transitions == -1)[0]
#
#             # 计算平均上升时间
#             for rise_idx in rising_indices:
#                 decay_idx = decaying_indices[decaying_indices > rise_idx]
#                 if len(decay_idx) > 0:
#                     decay_idx = decay_idx[0]
#                     rising_times.append(decay_idx - rise_idx)
#
#             # 计算平均下降时间
#             for decay_idx in decaying_indices:
#                 rise_idx = rising_indices[rising_indices < decay_idx]
#                 if len(rise_idx) > 0:
#                     rise_idx = rise_idx[-1]
#                     decaying_times.append(decay_idx - rise_idx)
#
#             rise_time_avg = np.mean(rising_times) if rising_times else 0
#             decay_time_avg = np.mean(decaying_times) if decaying_times else 0
#
#             # Latency 和 peak 信息 (可根据需求修改)
#             latency = rising_indices[0] if len(rising_indices) > 0 else 0
#             peak_value = amplitude  # 假设峰值为窗口的最大值
#
#             # # 初始聚类标签
#             # cluster_label = 0
#
#             # 添加结果
#             results.append({
#                 'Neuron': neuron,
#                 'Start Time': start_time,
#                 'Amplitude': amplitude,
#                 'Peak': peak_value,
#                 'Decay Time': decay_time_avg,
#                 'Rise Time': rise_time_avg,
#                 'Latency': latency,
#                 'Frequency': frequency,
#                 # 'Cluster': cluster_label
#             })
#
#     return pd.DataFrame(results)
#
#
# # 计算滑动窗口内的关键指标
# metrics_df = calculate_metrics_for_neurons(data, threshold, window_size, step_size)
#
# # 保存或展示计算结果
# metrics_df.to_csv("./data/Day3_Neuron_Calcium_Metrics.csv", index=False)  # 将结果保存为CSV文件
# print(metrics_df.head())  # 或打印前几行结果

# 按神经元顺序存储
import pandas as pd
import numpy as np

# 加载数据
file_path = './data/Day6.xlsx'  # 请替换为实际文件路径
data = pd.read_excel(file_path)

# 确保只处理数值列并忽略 FrameLost 列
neuron_columns = data.columns[1:]  # 假设 n1~n53 为神经元数据
data[neuron_columns] = data[neuron_columns].apply(pd.to_numeric, errors='coerce')

# 定义滑动窗口参数
window_size = 100
step_size = 10


# 定义计算关键指标的函数
def calculate_metrics_for_neurons(data, window_size, step_size):
    results = {neuron: [] for neuron in neuron_columns}  # 按神经元分组存储结果
    time_stamps = data['stamp']

    for start in range(0, len(data) - window_size + 1, step_size):
        window_end = start + window_size
        window_data = data.iloc[start:window_end]

        start_time = time_stamps[start]  # 获取窗口的开始时间戳

        for neuron in neuron_columns:
            neuron_signal = window_data[neuron]
            mean_value = neuron_signal.mean()  # 窗口内信号的平均值
            peak_value = neuron_signal.max()  # 窗口内信号的最大值

            # 1. Amplitude: 窗口内信号的最大值减去平均值
            amplitude = peak_value - mean_value

            # 2. Decay Time: 从峰值下降到一半的时间
            decay_time = np.argmax(neuron_signal <= peak_value / 2) if np.any(neuron_signal <= peak_value / 2) else len(
                neuron_signal)

            # 3. Rise Time: 从平均值上升到峰值的时间
            rise_time = np.argmax(neuron_signal >= mean_value)

            # 4. Latency: 上升时间与下降时间之和
            latency = decay_time + rise_time

            # 5. Frequency: 超过平均值的元素比例
            frequency = len(np.where(neuron_signal > mean_value)[0]) / len(neuron_signal)

            # 将结果存储到对应神经元的列表中
            results[neuron].append({
                'Start Time': start_time,
                'Amplitude': amplitude,
                'Peak': peak_value,
                'Decay Time': decay_time,
                'Rise Time': rise_time,
                'Latency': latency,
                'Frequency': frequency,
            })

    # 将每个神经元的数据转换为单独的 DataFrame，并合并
    all_neurons_data = []
    for neuron, neuron_data in results.items():
        neuron_df = pd.DataFrame(neuron_data)
        neuron_df.insert(0, 'Neuron', neuron)  # 插入神经元名称列
        all_neurons_data.append(neuron_df)

    # 合并所有神经元的数据
    return pd.concat(all_neurons_data, ignore_index=True)


# 计算滑动窗口内的关键指标
metrics_df = calculate_metrics_for_neurons(data, window_size, step_size)

# 保存或展示计算结果
metrics_df.to_excel("./data/Day6_Neuron_Calcium_Metrics.xlsx", index=False,sheet_name= 'Windows100_step10')  # 将结果保存为CSV文件
print(metrics_df.head())  # 或打印前几行结果
