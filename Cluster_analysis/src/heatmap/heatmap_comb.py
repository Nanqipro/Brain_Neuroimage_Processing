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

# 根据对应表准备数据
aligned_day3, aligned_day6, aligned_day9 = [], [], []
neuron_labels_day3, neuron_labels_day6, neuron_labels_day9 = [], [], []

for _, row in correspondence_table.iterrows():
    day3_neuron, day6_neuron, day9_neuron = row['Day3'], row['Day6'], row['Day9']

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
sns.heatmap(aligned_day3_df.T, cmap='RdYlBu', cbar=True, vmin=vmin, vmax=vmax)  # 转置 DataFrame 以调换横纵坐标
plt.title('Day3')
plt.xlabel('stamp')
plt.ylabel('neural')

# Day6 热图
plt.subplot(1, 3, 2)
sns.heatmap(aligned_day6_df.T, cmap='RdYlBu', cbar=True, vmin=vmin, vmax=vmax)  # 转置 DataFrame 以调换横纵坐标
plt.title('Day6')
plt.xlabel('stamp')
plt.ylabel('')

# Day9 热图
plt.subplot(1, 3, 3)
sns.heatmap(aligned_day9_df.T, cmap='RdYlBu', cbar=True, vmin=vmin, vmax=vmax)  # 转置 DataFrame 以调换横纵坐标
plt.title('Day9')
plt.xlabel('stamp')
plt.ylabel('')

plt.tight_layout()
plt.savefig('../../graph/heatmap_combined.png')
plt.close()
