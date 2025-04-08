# 有放入CD1的数据进行热图绘制
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 加载数据
day6_data = pd.read_excel('../../datasets/EMtrace.xlsx')

# 将 'stamp' 列设置为索引
day6_data = day6_data.set_index('stamp')

# 分离 'behavior' 列
frame_lost = day6_data['behavior']
day6_data = day6_data.drop(columns=['behavior'])

# 数据标准化（Z-score 标准化）
day6_data_standardized = (day6_data - day6_data.mean()) / day6_data.std()

# **步骤1：计算每个神经元信号峰值出现的时间点**

# 对于每个神经元，找到其信号达到最大值的时间戳
peak_times = day6_data_standardized.idxmax()

# **步骤2：按照峰值出现的时间对神经元进行排序**

# 将神经元按照峰值时间从早到晚排序
sorted_neurons = peak_times.sort_values().index

# **步骤3：重新排列数据**

# 根据排序后的神经元顺序重新排列 DataFrame 的列
sorted_day6_data = day6_data_standardized[sorted_neurons]

# **步骤4：找到所有 '放入CD1' 的时间点**

# 找到 'FrameLost' 列中值为 '放入CD1' 的索引位置
cd1_indices = frame_lost[frame_lost == 'CD1'].index

# **步骤5：绘制热图并标注所有事件**

# 设置绘图颜色范围
vmin, vmax = -2, 2  # 控制颜色对比度

# 绘制热图
plt.figure(figsize=(60, 15))
ax = sns.heatmap(sorted_day6_data.T, cmap='viridis', cbar=True, vmin=vmin, vmax=vmax)

# 如果找到了 '放入CD1' 的时间点，绘制垂直线并标注
for cd1_time in cd1_indices:
    # 检查 cd1_time 是否在 sorted_day6_data 的索引中
    if cd1_time in sorted_day6_data.index:
        # 获取对应的绘图位置
        cd1_position = sorted_day6_data.index.get_loc(cd1_time)
        # 绘制垂直线，颜色为亮黄色，线宽加大
        ax.axvline(x=cd1_position, color='white', linestyle='--', linewidth=3)
        # 添加文本标签，调整位置避免遮挡
        plt.text(cd1_position + 0.5, ax.get_ylim()[1] + 5, 'add CD1', color='yellow',
                 rotation=90, verticalalignment='bottom', fontsize=12)
        # 或者将标签放在图形顶部
        # plt.text(cd1_position + 0.5, -0.5, '放入CD1', color='yellow', rotation=90,
        #          verticalalignment='bottom', fontsize=12)

plt.title('EMtrace-heatmap', fontsize=16)
plt.xlabel('stamp', fontsize=20)
plt.ylabel('neuron', fontsize=20)

# 修改Y轴标签（神经元标签）的字体大小和粗细，设置为水平方向
ax.set_yticklabels(ax.get_yticklabels(), fontsize=14, fontweight='bold', rotation=0)

# 修改X轴标签（时间戳）的字体大小和粗细
ax.set_xticklabels(ax.get_xticklabels(), fontsize=14, fontweight='bold')

plt.savefig('../../graph/heatmap_sort_EMtrace.png')
plt.close()
