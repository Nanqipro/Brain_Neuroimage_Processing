def plot_trace_before_cd1(data, cd1_index, n_stamps, output_path, sampling_rate=4.8):
    """
    绘制CD1标签前n_stamps个时间戳的神经元组平均钙离子浓度轨迹图
    
    参数:
        data: 包含平均值和标准差的DataFrame
        cd1_index: CD1标签首次出现的索引
        n_stamps: 要绘制的时间戳数量
        output_path: 输出图像路径
        sampling_rate: 采样频率(Hz)
    """
    if cd1_index is None or cd1_index < n_stamps:
        print(f"警告: CD1标签前的数据不足{n_stamps}个时间戳，将使用所有可用数据")
        start_idx = 0
        end_idx = cd1_index - 1 if cd1_index is not None else len(data) - 1
    else:
        start_idx = cd1_index - n_stamps
        end_idx = cd1_index - 1
    
    # 提取相关数据段
    plot_data = data.iloc[start_idx:end_idx+1].copy()
    
    # 计算采样周期（秒）
    sampling_period = 1.0 / sampling_rate
    
    # 创建相对时间轴（以秒为单位，使用实际采样频率）
    plot_data['relative_time'] = [(i * sampling_period) for i in range(len(plot_data))]
    
    # 创建图像
    plt.figure(figsize=(12, 8))
    
    # 使用统一的颜色方案绘制三条曲线及其阴影区域
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
    
    # 添加图例和标签
    plt.legend(fontsize=12)
    plt.xlabel('Time (seconds)', fontsize=14)
    plt.ylabel('Δ F /F', fontsize=14)
    plt.title(f'Average Calcium Concentration of Neurons {n_stamps} Timestamps Before CD1', fontsize=16)
    
    # 设置统一的y轴范围
    plt.ylim([-0.3, 0.5])
    
    # 添加垂直线标记CD1出现时间点
    plt.axvline(x=plot_data['relative_time'].max(), color='k', linestyle='--', linewidth=3)
    plt.text(plot_data['relative_time'].max()-5, 0.35, 'CD1', fontsize=14)
    
    # 网格线
    plt.grid(False)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"CD1前轨迹图已保存至: {output_path}")
