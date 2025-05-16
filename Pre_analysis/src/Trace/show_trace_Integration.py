# 波动更明显
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 加载数据
day3_data = pd.read_excel('../datasets/Day3_with_behavior_labels_filled.xlsx')
day6_data = pd.read_excel('../datasets/Day6_with_behavior_labels_filled.xlsx')
day9_data = pd.read_excel('../datasets/Day9_with_behavior_labels_filled.xlsx')
correspondence_table = pd.read_excel('../datasets/神经元对应表.xlsx')

# 设置放大因子以增加波动幅度
amplitude_scale = 5  # 增大波动幅度的缩放系数，可以调整此值来增加或减少波动幅度

# 根据对应表准备数据
aligned_day3, aligned_day6, aligned_day9 = [], [], []
neuron_labels_day3, neuron_labels_day6, neuron_labels_day9 = [], [], []

for _, row in correspondence_table.iterrows():
    day3_neuron, day6_neuron, day9_neuron = row['Day3'], row['Day6'], row['Day9']

    # Day3 数据和标签；若为 null 则填充 NaN 并添加空标签
    if pd.notna(day3_neuron) and day3_neuron in day3_data.columns:
        trace = (day3_data[day3_neuron] - day3_data[day3_neuron].mean()) / day3_data[day3_neuron].std()
        aligned_day3.append(trace * amplitude_scale)
        neuron_labels_day3.append(day3_neuron)
    else:
        aligned_day3.append(pd.Series([np.nan] * len(day3_data)))
        neuron_labels_day3.append(None)

    # Day6 数据和标签；若为 null 则填充 NaN 并添加空标签
    if pd.notna(day6_neuron) and day6_neuron in day6_data.columns:
        trace = (day6_data[day6_neuron] - day6_data[day6_neuron].mean()) / day6_data[day6_neuron].std()
        aligned_day6.append(trace * amplitude_scale)
        neuron_labels_day6.append(day6_neuron)
    else:
        aligned_day6.append(pd.Series([np.nan] * len(day6_data)))
        neuron_labels_day6.append(None)

    # Day9 数据和标签；若为 null 则填充 NaN 并添加空标签
    if pd.notna(day9_neuron) and day9_neuron in day9_data.columns:
        trace = (day9_data[day9_neuron] - day9_data[day9_neuron].mean()) / day9_data[day9_neuron].std()
        aligned_day9.append(trace * amplitude_scale)
        neuron_labels_day9.append(day9_neuron)
    else:
        aligned_day9.append(pd.Series([np.nan] * len(day9_data)))
        neuron_labels_day9.append(None)

# 将列表转换为 DataFrame
aligned_day3_df = pd.DataFrame(aligned_day3).T
aligned_day6_df = pd.DataFrame(aligned_day6).T
aligned_day9_df = pd.DataFrame(aligned_day9).T

# 设置列名为神经元标签（包括 None 作为空白行占位）
aligned_day3_df.columns = neuron_labels_day3
aligned_day6_df.columns = neuron_labels_day6
aligned_day9_df.columns = neuron_labels_day9

# 设置纵向偏移量
offset = 20  # 增大每个神经元 trace 的垂直偏移量
time_stamps = day3_data['stamp']  # 使用时间戳作为横坐标
matplotlib.use('TkAgg') # 或者 'Agg'，'Qt5Agg'，根据你系统的支持情况
plt.figure(figsize=(40, 10))

# Day3 trace 图
plt.subplot(1, 3, 1)
for i, neuron_label in enumerate(neuron_labels_day3):
    if neuron_label is not None:
        plt.plot(time_stamps, aligned_day3_df[neuron_label] + i * offset)
plt.title('Day3')
plt.xlabel('Time Stamp')
plt.ylabel('Neuron Trace')
plt.yticks(np.arange(0, len(neuron_labels_day3) * offset, offset), neuron_labels_day3)

# Day6 trace 图
plt.subplot(1, 3, 2)
for i, neuron_label in enumerate(neuron_labels_day6):
    if neuron_label is not None:
        plt.plot(time_stamps, aligned_day6_df[neuron_label] + i * offset)
plt.title('Day6')
plt.xlabel('Time Stamp')
plt.yticks(np.arange(0, len(neuron_labels_day6) * offset, offset), neuron_labels_day6)
plt.ylabel('')

# Day9 trace 图
plt.subplot(1, 3, 3)
for i, neuron_label in enumerate(neuron_labels_day9):
    if neuron_label is not None:
        plt.plot(time_stamps, aligned_day9_df[neuron_label] + i * offset)
plt.title('Day9')
plt.xlabel('Time Stamp')
plt.yticks(np.arange(0, len(neuron_labels_day9) * offset, offset), neuron_labels_day9)
plt.ylabel('')

plt.tight_layout()
plt.show()