#colorful
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 加载数据
day3_data = pd.read_excel('../../datasets/Day3_with_behavior_labels_filled.xlsx')
day6_data = pd.read_excel('../../datasets/Day6_with_behavior_labels_filled.xlsx')
day9_data = pd.read_excel('../../datasets/Day9_with_behavior_labels_filled.xlsx')
correspondence_table = pd.read_excel('../../datasets/神经元对应表.xlsx')

# 首先创建Day3数据的标准化副本，用于计算峰值时间
day3_standardized = {}
for col in day3_data.columns:
    if col != 'stamp' and col != 'behavior':
        # 确保只处理神经元数据列
        series = day3_data[col]
        # 标准化数据
        day3_standardized[col] = (series - series.mean()) / series.std()

# 计算每个神经元在Day3中的峰值时间
day3_peak_times = {}
for neuron, data in day3_standardized.items():
    # 找出最大值所在的索引位置
    peak_idx = data.idxmax()
    day3_peak_times[neuron] = peak_idx

# 根据对应表准备数据
aligned_day3, aligned_day6, aligned_day9 = [], [], []
neuron_labels_day3, neuron_labels_day6, neuron_labels_day9 = [], [], []
peak_times_order = []  # 用于记录对应表中每行神经元的峰值时间

for _, row in correspondence_table.iterrows():
    day3_neuron, day6_neuron, day9_neuron = row['Day3'], row['Day6'], row['Day9']
    
    # 记录Day3神经元的峰值时间，如果存在的话
    peak_time = None
    if pd.notna(day3_neuron) and day3_neuron in day3_data.columns:
        if day3_neuron in day3_peak_times:
            peak_time = day3_peak_times[day3_neuron]
    peak_times_order.append((peak_time, _))  # 保存峰值时间和行索引

    # 添加 Day3 的数据和标签；若为 null 则填充 NaN
    if pd.notna(day3_neuron) and day3_neuron in day3_data.columns:
        aligned_day3.append(day3_data[day3_neuron])
        neuron_labels_day3.append(day3_neuron)
    else:
        aligned_day3.append(pd.Series([np.nan] * len(day3_data)))
        neuron_labels_day3.append(None)

    # 添加 Day6 的数据和标签；若为 null 则填充 NaN
    if pd.notna(day6_neuron) and day6_neuron in day6_data.columns:
        aligned_day6.append(day6_data[day6_neuron])
        neuron_labels_day6.append(day6_neuron)
    else:
        aligned_day6.append(pd.Series([np.nan] * len(day6_data)))
        neuron_labels_day6.append(None)

    # 添加 Day9 的数据和标签；若为 null 则填充 NaN
    if pd.notna(day9_neuron) and day9_neuron in day9_data.columns:
        aligned_day9.append(day9_data[day9_neuron])
        neuron_labels_day9.append(day9_neuron)
    else:
        aligned_day9.append(pd.Series([np.nan] * len(day9_data)))
        neuron_labels_day9.append(None)

# 按照Day3的峰值时间排序（先去除None值的行再排序）
valid_peak_times = [(t, i) for t, i in peak_times_order if t is not None]
sorted_indices = [i for _, i in sorted(valid_peak_times, key=lambda x: x[0])]
# 将None值的行添加到最后
none_indices = [i for t, i in peak_times_order if t is None]
sorted_indices.extend(none_indices)

# 根据排序后的索引重新排列数据
aligned_day3 = [aligned_day3[i] for i in sorted_indices]
aligned_day6 = [aligned_day6[i] for i in sorted_indices]
aligned_day9 = [aligned_day9[i] for i in sorted_indices]
neuron_labels_day3 = [neuron_labels_day3[i] for i in sorted_indices]
neuron_labels_day6 = [neuron_labels_day6[i] for i in sorted_indices]
neuron_labels_day9 = [neuron_labels_day9[i] for i in sorted_indices]

# 将列表转换为 DataFrame，并设置索引为时间戳 (stamp)，再转置以交换横纵坐标
aligned_day3_df = pd.DataFrame(aligned_day3, index=neuron_labels_day3).T
aligned_day6_df = pd.DataFrame(aligned_day6, index=neuron_labels_day6).T
aligned_day9_df = pd.DataFrame(aligned_day9, index=neuron_labels_day9).T

# 标准化数据
aligned_day3_df = (aligned_day3_df - aligned_day3_df.mean()) / aligned_day3_df.std()
aligned_day6_df = (aligned_day6_df - aligned_day6_df.mean()) / aligned_day6_df.std()
aligned_day9_df = (aligned_day9_df - aligned_day9_df.mean()) / aligned_day9_df.std()

# 设置绘图颜色范围
vmin, vmax = -2, 2  # 控制颜色对比度

# 绘制热图
plt.figure(figsize=(40, 10))

# Day3 热图
plt.subplot(1, 3, 1)
ax1 = sns.heatmap(aligned_day3_df.T, cmap='viridis', cbar=True, vmin=vmin, vmax=vmax)  # 转置 DataFrame 以调换横纵坐标
plt.title('Day3 (Sorted by Peak Time)', fontsize=25)
plt.xlabel('stamp', fontsize=25)
plt.ylabel('neuron', fontsize=25)

# Day6 热图
plt.subplot(1, 3, 2)
ax2 = sns.heatmap(aligned_day6_df.T, cmap='viridis', cbar=True, vmin=vmin, vmax=vmax)  # 转置 DataFrame 以调换横纵坐标
plt.title('Day6 (Using Day3 Sorting)', fontsize=25)
plt.xlabel('stamp', fontsize=25)
plt.ylabel('')

# Day9 热图
plt.subplot(1, 3, 3)
ax3 = sns.heatmap(aligned_day9_df.T, cmap='viridis', cbar=True, vmin=vmin, vmax=vmax)  # 转置 DataFrame 以调换横纵坐标
plt.title('Day9 (Using Day3 Sorting)', fontsize=25)
plt.xlabel('stamp', fontsize=25)
plt.ylabel('')

plt.tight_layout()
plt.savefig('../../graph/heatmap_combined_sorted.png')
plt.close()
