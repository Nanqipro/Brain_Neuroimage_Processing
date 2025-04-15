# 有放入CD1的数据进行热图绘制
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas

# 加载数据
day6_data = pd.read_excel('../../datasets/No.297920240925_104733trace.xlsx')

# 将 'stamp' 列设置为索引
day6_data = day6_data.set_index('stamp')

# 检查是否存在 'behavior' 列
has_behavior = 'behavior' in day6_data.columns

# 分离 'behavior' 列（如果存在）
if has_behavior:
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

# **步骤4：找到所有行为标签的首次出现时间点**

# 初始化行为标记变量
behavior_indices = {}
unique_behaviors = []

# 只有当behavior列存在时才处理行为标签
if has_behavior:
    # 获取所有不同的行为标签
    unique_behaviors = frame_lost.dropna().unique()
    
    # 对frame_lost进行处理，找出每种行为连续出现时的第一个时间点
    previous_behavior = None
    for timestamp, behavior in frame_lost.items():
        # 跳过空值
        if pd.isna(behavior):
            continue
        
        # 如果与前一个行为不同，则记录该时间点
        if behavior != previous_behavior:
            if behavior not in behavior_indices:
                behavior_indices[behavior] = []
            behavior_indices[behavior].append(timestamp)
        
        previous_behavior = behavior

# **步骤5：绘制热图并标注所有事件**

# 设置绘图颜色范围
vmin, vmax = -2, 2  # 控制颜色对比度

# 创建图形和轴，减少默认边距
fig = plt.figure(figsize=(25, 15))
# 调整子图位置，减少边距
plt.subplots_adjust(left=0.05, right=0.98, top=0.9, bottom=0.15)

# 绘制热图
ax = sns.heatmap(sorted_day6_data.T, cmap='viridis', cbar=True, vmin=vmin, vmax=vmax)

# 只有当behavior列存在时才添加行为标记
if has_behavior and unique_behaviors:
    # 颜色映射，为每种行为分配不同的颜色
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_behaviors)))
    color_map = {behavior: colors[i] for i, behavior in enumerate(unique_behaviors)}

    # 为每种行为绘制垂直线并标注
    for behavior, timestamps in behavior_indices.items():
        for behavior_time in timestamps:
            # 检查行为时间是否在排序后的数据索引中
            if behavior_time in sorted_day6_data.index:
                # 获取对应的绘图位置
                position = sorted_day6_data.index.get_loc(behavior_time)
                # 绘制垂直线，白色虚线
                ax.axvline(x=position, color='white', linestyle='--', linewidth=2)
                # 添加文本标签，放在热图外部并使用黑色字体
                plt.text(position + 0.5, -5, behavior, 
                        color='black', rotation=90, verticalalignment='top', fontsize=12, fontweight='bold')

plt.title('No.297920240925_104733trace-heatmap', fontsize=16)
plt.xlabel('stamp', fontsize=20)
plt.ylabel('neuron', fontsize=20)

# 修改Y轴标签（神经元标签）的字体大小和粗细，设置为水平方向
ax.set_yticklabels(ax.get_yticklabels(), fontsize=14, fontweight='bold', rotation=0)

# 修改X轴标签（时间戳）的字体大小和粗细
ax.set_xticklabels(ax.get_xticklabels(), fontsize=14, fontweight='bold')

# 应用紧凑布局
plt.tight_layout()

# 保存图像时使用紧凑边界设置
plt.savefig('../../graph/heatmap_sort_No.297920240925_104733trace.png', bbox_inches='tight', pad_inches=0.1, dpi=100)
plt.close()
