# 这是一个展示如何在plot_trace_before_cd1和plot_trace_after_cd1函数中添加阴影区域的示例

# 在plot_trace_before_cd1函数中，找到以下代码段（约230-242行）：
# -------------------------------------------------------------
# Group 1 - 绿色
plt.plot(plot_data['relative_time'], plot_data['group1_avg'], 
        color=GROUP_COLORS['group1'], label='Group 1', linewidth=4)
# 阴影区域显示已移除

# Group 2 - 黄色
plt.plot(plot_data['relative_time'], plot_data['group2_avg'], 
        color=GROUP_COLORS['group2'], label='Group 2', linewidth=4)
# 阴影区域显示已移除

# Group 3 - 灰色虚线
plt.plot(plot_data['relative_time'], plot_data['group3_avg'], 
        color=GROUP_COLORS['group3'], label='Group 3', linewidth=4, linestyle='--')
# 阴影区域显示已移除
# -------------------------------------------------------------

# 替换为以下代码：
# -------------------------------------------------------------
# Group 1 - 绿色
plt.plot(plot_data['relative_time'], plot_data['group1_avg'], 
        color=GROUP_COLORS['group1'], label='Group 1', linewidth=4)
# 添加阴影区域表示标准差
plt.fill_between(plot_data['relative_time'], 
                plot_data['group1_avg'] - plot_data['group1_std'], 
                plot_data['group1_avg'] + plot_data['group1_std'], 
                color=GROUP_COLORS['group1'], alpha=0.3)

# Group 2 - 黄色
plt.plot(plot_data['relative_time'], plot_data['group2_avg'], 
        color=GROUP_COLORS['group2'], label='Group 2', linewidth=4)
# 添加阴影区域表示标准差
plt.fill_between(plot_data['relative_time'], 
                plot_data['group2_avg'] - plot_data['group2_std'], 
                plot_data['group2_avg'] + plot_data['group2_std'], 
                color=GROUP_COLORS['group2'], alpha=0.3)

# Group 3 - 灰色虚线
plt.plot(plot_data['relative_time'], plot_data['group3_avg'], 
        color=GROUP_COLORS['group3'], label='Group 3', linewidth=4, linestyle='--')
# 添加阴影区域表示标准差
plt.fill_between(plot_data['relative_time'], 
                plot_data['group3_avg'] - plot_data['group3_std'], 
                plot_data['group3_avg'] + plot_data['group3_std'], 
                color=GROUP_COLORS['group3'], alpha=0.3)
# -------------------------------------------------------------

# 在plot_trace_after_cd1函数中，类似地修改（约300-315行）：
# -------------------------------------------------------------
# Group 1 - 绿色
plt.plot(plot_data['relative_time'], plot_data['group1_avg'], 
        color=GROUP_COLORS['group1'], label='Group 1', linewidth=4)
# 阴影区域显示已移除

# Group 2 - 黄色
plt.plot(plot_data['relative_time'], plot_data['group2_avg'], 
        color=GROUP_COLORS['group2'], label='Group 2', linewidth=4)
# 阴影区域显示已移除

# Group 3 - 灰色虚线
plt.plot(plot_data['relative_time'], plot_data['group3_avg'], 
        color=GROUP_COLORS['group3'], label='Group 3', linewidth=4, linestyle='--')
# 阴影区域显示已移除
# -------------------------------------------------------------

# 替换为以下代码：
# -------------------------------------------------------------
# Group 1 - 绿色
plt.plot(plot_data['relative_time'], plot_data['group1_avg'], 
        color=GROUP_COLORS['group1'], label='Group 1', linewidth=4)
# 添加阴影区域表示标准差
plt.fill_between(plot_data['relative_time'], 
                plot_data['group1_avg'] - plot_data['group1_std'], 
                plot_data['group1_avg'] + plot_data['group1_std'], 
                color=GROUP_COLORS['group1'], alpha=0.3)

# Group 2 - 黄色
plt.plot(plot_data['relative_time'], plot_data['group2_avg'], 
        color=GROUP_COLORS['group2'], label='Group 2', linewidth=4)
# 添加阴影区域表示标准差
plt.fill_between(plot_data['relative_time'], 
                plot_data['group2_avg'] - plot_data['group2_std'], 
                plot_data['group2_avg'] + plot_data['group2_std'], 
                color=GROUP_COLORS['group2'], alpha=0.3)

# Group 3 - 灰色虚线
plt.plot(plot_data['relative_time'], plot_data['group3_avg'], 
        color=GROUP_COLORS['group3'], label='Group 3', linewidth=4, linestyle='--')
# 添加阴影区域表示标准差
plt.fill_between(plot_data['relative_time'], 
                plot_data['group3_avg'] - plot_data['group3_std'], 
                plot_data['group3_avg'] + plot_data['group3_std'], 
                color=GROUP_COLORS['group3'], alpha=0.3)
# -------------------------------------------------------------
