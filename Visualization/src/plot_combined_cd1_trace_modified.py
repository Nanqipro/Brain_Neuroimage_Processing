def plot_combined_cd1_trace(data, cd1_index, before_stamps, after_stamps, output_path, sampling_rate=4.8):
    """
    在同一个图中绘制CD1标签前后的神经元组平均钙离子浓度轨迹图
    
    参数:
        data: 包含平均值和标准差的DataFrame
        cd1_index: CD1标签首次出现的索引
        before_stamps: CD1标签前的时间戳数量
        after_stamps: CD1标签后的时间戳数量
        output_path: 输出图像路径
        sampling_rate: 采样频率(Hz)
    """
    if cd1_index is None:
        print("警告: 未找到CD1标签，无法绘制组合图表")
        return
    
    # 处理CD1前的数据
    if cd1_index < before_stamps:
        print(f"警告: CD1标签前的数据不足{before_stamps}个时间戳，将使用所有可用数据")
        before_start_idx = 0
    else:
        before_start_idx = cd1_index - before_stamps
    
    before_end_idx = cd1_index - 1
    
    # 处理CD1后的数据
    if cd1_index + after_stamps > len(data):
        print(f"警告: CD1标签后的数据不足{after_stamps}个时间戳，将使用所有可用数据")
        after_end_idx = len(data) - 1
    else:
        after_end_idx = cd1_index + after_stamps - 1
    
    # 提取相关数据段
    before_data = data.iloc[before_start_idx:before_end_idx+1].copy()
    after_data = data.iloc[cd1_index:after_end_idx+1].copy()
    
    # 计算采样周期（秒）
    sampling_period = 1.0 / sampling_rate
    
    # 创建相对时间轴（以秒为单位，使用实际采样频率）
    # CD1前的时间为负值，CD1时刻为0，CD1后为正值
    before_data['relative_time'] = [((i - len(before_data)) * sampling_period) for i in range(len(before_data))]
    after_data['relative_time'] = [(i * sampling_period) for i in range(len(after_data))]
    
    # 创建图像，增加图表大小以解决布局问题
    plt.figure(figsize=(15, 10))
    
    # 绘制CD1前的轨迹
    # Group 1 - 绿色
    plt.plot(before_data['relative_time'], before_data['group1_avg'], 
             color=GROUP_COLORS['group1'], label='Group 1', linewidth=4)
    # 添加阴影区域表示标准差
    plt.fill_between(before_data['relative_time'], 
                    before_data['group1_avg'] - before_data['group1_std'], 
                    before_data['group1_avg'] + before_data['group1_std'], 
                    color=GROUP_COLORS['group1'], alpha=0.3)
    
    # Group 2 - 黄色
    plt.plot(before_data['relative_time'], before_data['group2_avg'], 
             color=GROUP_COLORS['group2'], label='Group 2', linewidth=4)
    # 添加阴影区域表示标准差
    plt.fill_between(before_data['relative_time'], 
                    before_data['group2_avg'] - before_data['group2_std'], 
                    before_data['group2_avg'] + before_data['group2_std'], 
                    color=GROUP_COLORS['group2'], alpha=0.3)
    
    # Group 3 - 灰色虚线
    plt.plot(before_data['relative_time'], before_data['group3_avg'], 
             color=GROUP_COLORS['group3'], label='Group 3', linewidth=4, linestyle='--')
    # 添加阴影区域表示标准差
    plt.fill_between(before_data['relative_time'], 
                    before_data['group3_avg'] - before_data['group3_std'], 
                    before_data['group3_avg'] + before_data['group3_std'], 
                    color=GROUP_COLORS['group3'], alpha=0.3)
    
    # 绘制CD1后的轨迹
    # Group 1 - 绿色
    plt.plot(after_data['relative_time'], after_data['group1_avg'], 
             color=GROUP_COLORS['group1'], linewidth=4)
    # 添加阴影区域表示标准差
    plt.fill_between(after_data['relative_time'], 
                    after_data['group1_avg'] - after_data['group1_std'], 
                    after_data['group1_avg'] + after_data['group1_std'], 
                    color=GROUP_COLORS['group1'], alpha=0.3)
    
    # Group 2 - 黄色
    plt.plot(after_data['relative_time'], after_data['group2_avg'], 
             color=GROUP_COLORS['group2'], linewidth=4)
    # 添加阴影区域表示标准差
    plt.fill_between(after_data['relative_time'], 
                    after_data['group2_avg'] - after_data['group2_std'], 
                    after_data['group2_avg'] + after_data['group2_std'], 
                    color=GROUP_COLORS['group2'], alpha=0.3)
    
    # Group 3 - 灰色虚线
    plt.plot(after_data['relative_time'], after_data['group3_avg'], 
             color=GROUP_COLORS['group3'], linewidth=4, linestyle='--')
    # 添加阴影区域表示标准差
    plt.fill_between(after_data['relative_time'], 
                    after_data['group3_avg'] - after_data['group3_std'], 
                    after_data['group3_avg'] + after_data['group3_std'], 
                    color=GROUP_COLORS['group3'], alpha=0.3)
    
    # 添加图例和标签
    plt.legend(fontsize=12)
    plt.xlabel('Time (seconds)', fontsize=14)
    plt.ylabel('Δ F /F', fontsize=14)
    plt.title(f'Comparison of Average Calcium Concentration of Neurons Before and After CD1 ({before_stamps}/{after_stamps} timestamps)', fontsize=16)
    
    # 设置统一的y轴范围
    plt.ylim([-0.3, 0.5])
    
    # 添加垂直线标记CD1出现时间点
    plt.axvline(x=0, color='k', linestyle='--', linewidth=3)
    plt.text(0.05, 0.35, 'CD1', fontsize=14)
    
    # 在图中标记CD1前后
    plt.text(-before_stamps*sampling_period*0.5, 0.3, 'Before CD1', fontsize=12, ha='center')
    plt.text(after_stamps*sampling_period*0.5, 0.3, 'After CD1', fontsize=12, ha='center')
    
    # 网格线
    plt.grid(False)
    
    # 设置更大的边距以避免警告
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    # 保存图像
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"CD1前后对比轨迹图已保存至: {output_path}")
