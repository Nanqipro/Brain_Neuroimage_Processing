# # 无放入CD1的数据进行热图绘制
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# # 加载数据
# day3_data = pd.read_excel('./data/Day3.xlsx')
#
# # 将 'stamp' 列设置为索引
# day3_data = day3_data.set_index('stamp')
#
# # 数据标准化（Z-score 标准化）
# day3_data_standardized = (day3_data - day3_data.mean()) / day3_data.std()
#
# # **步骤1：计算每个神经元信号峰值出现的时间点**
#
# # 对于每个神经元，找到其信号达到最大值的时间戳
# peak_times = day3_data_standardized.idxmax()
#
# # **步骤2：按照峰值出现的时间对神经元进行排序**
#
# # 将神经元按照峰值时间从早到晚排序
# sorted_neurons = peak_times.sort_values().index
#
# # **步骤3：重新排列数据**
#
# # 根据排序后的神经元顺序重新排列 DataFrame 的列
# sorted_day3_data = day3_data_standardized[sorted_neurons]
#
# # **步骤4：绘制热图**
#
# # 设置绘图颜色范围
# vmin, vmax = -2, 2  # 控制颜色对比度
#
# # 绘制热图
# plt.figure(figsize=(20, 10))
# sns.heatmap(sorted_day3_data.T, cmap='RdYlBu', cbar=True, vmin=vmin, vmax=vmax)
# plt.title('Day3-heatmap')
# plt.xlabel('stamp')
# plt.ylabel('neuron')
# plt.tight_layout()
# plt.show()

# 有放入CD1的数据进行热图绘制
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 加载数据
day3_data = pd.read_excel('./data/Day9_with_behavior_labels_filled.xlsx')

# 将 'stamp' 列设置为索引
day3_data = day3_data.set_index('stamp')

# 分离 'FrameLost' 列
frame_lost = day3_data['FrameLost']
day3_data = day3_data.drop(columns=['FrameLost'])

# 数据标准化（Z-score 标准化）
day3_data_standardized = (day3_data - day3_data.mean()) / day3_data.std()

# **步骤1：计算每个神经元信号峰值出现的时间点**

# 对于每个神经元，找到其信号达到最大值的时间戳
peak_times = day3_data_standardized.idxmax()

# **步骤2：按照峰值出现的时间对神经元进行排序**

# 将神经元按照峰值时间从早到晚排序
sorted_neurons = peak_times.sort_values().index

# **步骤3：重新排列数据**

# 根据排序后的神经元顺序重新排列 DataFrame 的列
sorted_day3_data = day3_data_standardized[sorted_neurons]

# **步骤4：找到所有 '放入CD1' 的时间点**

# 找到 'FrameLost' 列中值为 '放入CD1' 的索引位置
cd1_indices = frame_lost[frame_lost == '放入CD1'].index

# **步骤5：绘制热图并标注所有事件**

# 设置绘图颜色范围
vmin, vmax = -2, 2  # 控制颜色对比度

# 绘制热图
plt.figure(figsize=(25, 15))
ax = sns.heatmap(sorted_day3_data.T, cmap='viridis', cbar=True, vmin=vmin, vmax=vmax)

# 如果找到了 '放入CD1' 的时间点，绘制垂直线并标注
for cd1_time in cd1_indices:
    # 检查 cd1_time 是否在 sorted_day3_data 的索引中
    if cd1_time in sorted_day3_data.index:
        # 获取对应的绘图位置
        cd1_position = sorted_day3_data.index.get_loc(cd1_time)
        # 绘制垂直线，颜色为亮黄色，线宽加大
        ax.axvline(x=cd1_position, color='white', linestyle='--', linewidth=3)
        # 添加文本标签，调整位置避免遮挡
        plt.text(cd1_position + 0.5, ax.get_ylim()[1] + 5, 'add CD1', color='yellow',
                 rotation=90, verticalalignment='bottom', fontsize=12)
        # 或者将标签放在图形顶部
        # plt.text(cd1_position + 0.5, -0.5, '放入CD1', color='yellow', rotation=90,
        #          verticalalignment='bottom', fontsize=12)

plt.title('Day9-heatmap (add CD1)', fontsize=16)
plt.xlabel('stamp', fontsize=20)
plt.ylabel('neuron', fontsize=20)

# 修改Y轴标签（神经元标签）的字体大小和粗细
ax.set_yticklabels(ax.get_yticklabels(), fontsize=14, fontweight='bold')

# 修改X轴标签（时间戳）的字体大小和粗细
ax.set_xticklabels(ax.get_xticklabels(), fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()
