import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from matplotlib.patches import Patch

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='绘制神经元分组的钙离子浓度轨迹图')
    parser.add_argument('--input', type=str, 
                        default='../datasets/Day3_with_behavior_labels_filled.xlsx',
                        help='输入数据文件路径')
    parser.add_argument('--output-dir', type=str, 
                        default='../results/CD1_traces_day3/',
                        help='输出图像目录')
    parser.add_argument('--position-file', type=str,
                        default='../datasets/Day3_Max_position.csv',
                        help='神经元位置坐标文件路径')
    parser.add_argument('--before-stamps', type=int, default=100,
                        help='CD1标签前的时间戳数量')
    parser.add_argument('--after-stamps', type=int, default=100,
                        help='CD1标签后的时间戳数量')
    parser.add_argument('--no-shadow', action='store_true',
                        help='不显示标准差阴影区域')
    parser.add_argument('--sampling-rate', type=float, default=4.8,
                        help='采样频率(Hz)，默认为4.8Hz')
    return parser.parse_args()

def ensure_dir(directory):
    """
    确保输出目录存在
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_data(file_path):
    """
    加载Excel数据文件
    
    参数:
        file_path: Excel文件路径
        
    返回:
        data: 包含数据的DataFrame
    """
    print(f"正在加载数据: {file_path}")
    data = pd.read_excel(file_path)
    
    # 检查数据结构
    if 'stamp' not in data.columns:
        raise ValueError("数据中缺少'stamp'列")
    
    # 检查是否有behavior列
    if 'behavior' not in data.columns:
        raise ValueError("数据中缺少'behavior'列，无法定位CD1标签")
    
    print(f"数据加载完成，共 {len(data)} 行, {len(data.columns)} 列")
    return data

def group_neurons(data):
    """
    将神经元分为三类
    
    参数:
        data: 包含神经元数据的DataFrame
        
    返回:
        group1_cols: 第一组神经元列名
        group2_cols: 第二组神经元列名
        group3_cols: 第三组神经元列名
    """
    # #按Day3分组
    # 定义神经元分组 day3
    group1 = ['n49', 'n32', 'n22', 'n47', 'n17', 'n12', 'n28', 'n18', 'n42', 
              'n46', 'n38', 'n40', 'n52', 'n7', 'n45', 'n43', 'n15', 'n16', 'n2', 'n51', 'n53']
    
    group2 = ['n20', 'n33', 'n21', 'n5', 'n1', 'n26', 'n25', 'n36', 'n44', 
              'n34', 'n4', 'n24', 'n10', 'n11', 'n9', 'n3', 'n41', 'n35', 'n48', 'n19']
    
    # # 定义神经元分组 day6
    # group1 = ['n2', 'n10', 'n15', 'n17', 'n18', 'n22', 'n40', 'n24', 'n27', 
    #           'n39', 'n25', 'n59', 'n42', 'n38', 'n50', 'n51', 'n46', 'n52']
    
    # group2 = ['n3', 'n4', 'n34', 'n12', 'n13', 'n14', 'n21', 'n29', 'n31', 
    #           'n4', 'n57', 'n5', 'n32', 'n60', 'n41']

    # # 定义神经元分组 day9
    # group1 = ['n38', 'n9', 'n17', 'n31', 'n23', 'n20', 'n22', 'n26', 'n12', 
    #           'n35', 'n42', 'n53', 'n46', 'n23', 'n51', 'n40', 'n1', 'n57']
    
    # group2 = ['n2', 'n6', 'n3', 'n7', 'n11', 'n13', 'n29', 'n24', 'n39', 
    #           'n10', 'n58', 'n47', 'n5', 'n61', 'n54', 'n18']

    # 按Day6分组
    # #day3
    # # 定义神经元分组
    # group1 = ['n5', 'n10', 'n11', 'n18', 'n21', 'n29', 'n34', 'n36', 'n43', 
    #           'n44', 'n45']
    
    # group2 = ['n14', 'n17', 'n19', 'n20', 'n32', 'n50']


    # #day6
    # # 定义神经元分组
    # group1 = ['n32', 'n31', 'n48', 'n28', 'n36', 'n40', 'n35', 'n34', 'n33', 
    #           'n38', 'n14', 'n49', 'n13', 'n41', 'n5', 'n26','n50']
    
    # group2 = ['n47', 'n62', 'n23', 'n22', 'n39', 'n61', 'n16', 'n6', 'n44', 
    #           'n21', 'n45', 'n29']


    # #day9
    # # 定义神经元分组
    # group1 = ['n7', 'n11', 'n13', 'n22', 'n39', 'n37', 'n5', 'n18']
    
    # group2 = ['n19', 'n20', 'n29', 'n24', 'n35', 'n2']
    
    
    # 获取所有神经元列名（排除'stamp'和'behavior'列）
    all_neurons = [col for col in data.columns if col not in ['stamp', 'behavior']]
    
    # 确认组1神经元存在于数据中
    group1_cols = [col for col in group1 if col in all_neurons]
    if len(group1_cols) != len(group1):
        missing = set(group1) - set(group1_cols)
        print(f"警告: 组1中有 {len(missing)} 个神经元不在数据中: {missing}")
    
    # 确认组2神经元存在于数据中
    group2_cols = [col for col in group2 if col in all_neurons]
    if len(group2_cols) != len(group2):
        missing = set(group2) - set(group2_cols)
        print(f"警告: 组2中有 {len(missing)} 个神经元不在数据中: {missing}")
    
    # 第三组为剩余的所有神经元
    group3_cols = [col for col in all_neurons if col not in group1_cols and col not in group2_cols]
    
    print(f"神经元分组完成:")
    print(f"  - 组1: {len(group1_cols)} 个神经元")
    print(f"  - 组2: {len(group2_cols)} 个神经元")
    print(f"  - 组3: {len(group3_cols)} 个神经元")
    
    return group1_cols, group2_cols, group3_cols

def find_cd1_index(data):
    """
    找到CD1标签首次出现的索引
    
    参数:
        data: 包含行为数据的DataFrame
        
    返回:
        cd1_index: CD1标签首次出现的索引，如果不存在则返回None
    """
    # 查找behavior列中首次出现'CD1'的位置
    cd1_rows = data[data['behavior'] == 'CD1']
    
    if cd1_rows.empty:
        print("警告: 数据中没有找到'CD1'标签")
        return None
    
    # 获取第一个CD1标签的索引
    cd1_index = cd1_rows.index[0]
    print(f"找到CD1标签首次出现位置: 索引 {cd1_index}, stamp {data.loc[cd1_index, 'stamp']}")
    
    return cd1_index

def calculate_group_averages(data, group1_cols, group2_cols, group3_cols):
    """
    计算每组神经元的平均钙离子浓度及标准差
    
    参数:
        data: 数据DataFrame
        group1_cols, group2_cols, group3_cols: 各组神经元列名
        
    返回:
        data_with_avgs: 添加了平均值列的DataFrame
    """
    # 为每组计算平均值
    data['group1_avg'] = data[group1_cols].mean(axis=1)
    data['group2_avg'] = data[group2_cols].mean(axis=1)
    data['group3_avg'] = data[group3_cols].mean(axis=1)
    
    # 计算每组的标准差，用于绘制阴影区域
    data['group1_std'] = data[group1_cols].std(axis=1)
    data['group2_std'] = data[group2_cols].std(axis=1)
    data['group3_std'] = data[group3_cols].std(axis=1)
    
    return data

# 定义统一的颜色方案
GROUP_COLORS = {
    'group1': '#4DAF4A',  # 绿色
    'group2': '#FFD700',  # 黄色
    'group3': '#000000'   # 黑色
}

def plot_trace_before_cd1(data, cd1_index, n_stamps, output_path, show_shadow=True, sampling_rate=4.8):
    """
    绘制CD1标签前n_stamps个时间戳的神经元组平均钙离子浓度轨迹图
    
    参数:
        data: 包含平均值和标准差的DataFrame
        cd1_index: CD1标签首次出现的索引
        n_stamps: 要绘制的时间戳数量
        output_path: 输出图像路径
        show_shadow: 是否显示标准差阴影区域
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
             color=GROUP_COLORS['group1'], label='Group 1', linewidth=2)
    if show_shadow:
        plt.fill_between(plot_data['relative_time'], 
                         plot_data['group1_avg'] - plot_data['group1_std'],
                         plot_data['group1_avg'] + plot_data['group1_std'],
                         color=GROUP_COLORS['group1'], alpha=0.2)
    
    # Group 2 - 黄色
    plt.plot(plot_data['relative_time'], plot_data['group2_avg'], 
             color=GROUP_COLORS['group2'], label='Group 2', linewidth=2)
    if show_shadow:
        plt.fill_between(plot_data['relative_time'], 
                         plot_data['group2_avg'] - plot_data['group2_std'],
                         plot_data['group2_avg'] + plot_data['group2_std'],
                         color=GROUP_COLORS['group2'], alpha=0.2)
    
    # Group 3 - 黑色
    plt.plot(plot_data['relative_time'], plot_data['group3_avg'], 
             color=GROUP_COLORS['group3'], label='Group 3', linewidth=2)
    if show_shadow:
        plt.fill_between(plot_data['relative_time'], 
                         plot_data['group3_avg'] - plot_data['group3_std'],
                         plot_data['group3_avg'] + plot_data['group3_std'],
                         color=GROUP_COLORS['group3'], alpha=0.2)
    
    # 添加图例和标签
    plt.legend(fontsize=12)
    plt.xlabel('Time (seconds)', fontsize=14)
    plt.ylabel('Δ F /F', fontsize=14)
    plt.title(f'Average Calcium Concentration of Neurons {n_stamps} Timestamps Before CD1', fontsize=16)
    
    # 设置统一的y轴范围
    plt.ylim([-0.3, 0.5])
    
    # 添加垂直线标记CD1出现时间点
    plt.axvline(x=plot_data['relative_time'].max(), color='k', linestyle='--', linewidth=2)
    plt.text(plot_data['relative_time'].max()-5, 0.35, 'CD1', fontsize=14)
    
    # 网格线
    plt.grid(False)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"CD1前轨迹图已保存至: {output_path}")

def plot_trace_after_cd1(data, cd1_index, n_stamps, output_path, show_shadow=True, sampling_rate=4.8):
    """
    绘制CD1标签后n_stamps个时间戳的神经元组平均钙离子浓度轨迹图
    
    参数:
        data: 包含平均值和标准差的DataFrame
        cd1_index: CD1标签首次出现的索引
        n_stamps: 要绘制的时间戳数量
        output_path: 输出图像路径
        show_shadow: 是否显示标准差阴影区域
        sampling_rate: 采样频率(Hz)
    """
    if cd1_index is None:
        print("警告: 未找到CD1标签，无法绘制CD1后的图表")
        return
    
    if cd1_index + n_stamps > len(data):
        print(f"警告: CD1标签后的数据不足{n_stamps}个时间戳，将使用所有可用数据")
        end_idx = len(data) - 1
    else:
        end_idx = cd1_index + n_stamps - 1
    
    # 提取相关数据段
    plot_data = data.iloc[cd1_index:end_idx+1].copy()
    
    # 计算采样周期（秒）
    sampling_period = 1.0 / sampling_rate
    
    # 创建相对时间轴（以秒为单位，使用实际采样频率）
    plot_data['relative_time'] = [(i * sampling_period) for i in range(len(plot_data))]
    
    # 创建图像
    plt.figure(figsize=(12, 8))
    
    # 使用统一的颜色方案绘制三条曲线及其阴影区域
    # Group 1 - 绿色
    plt.plot(plot_data['relative_time'], plot_data['group1_avg'], 
             color=GROUP_COLORS['group1'], label='Group 1', linewidth=2)
    if show_shadow:
        plt.fill_between(plot_data['relative_time'], 
                         plot_data['group1_avg'] - plot_data['group1_std'],
                         plot_data['group1_avg'] + plot_data['group1_std'],
                         color=GROUP_COLORS['group1'], alpha=0.2)
    
    # Group 2 - 黄色
    plt.plot(plot_data['relative_time'], plot_data['group2_avg'], 
             color=GROUP_COLORS['group2'], label='Group 2', linewidth=2)
    if show_shadow:
        plt.fill_between(plot_data['relative_time'], 
                         plot_data['group2_avg'] - plot_data['group2_std'],
                         plot_data['group2_avg'] + plot_data['group2_std'],
                         color=GROUP_COLORS['group2'], alpha=0.2)
    
    # Group 3 - 黑色
    plt.plot(plot_data['relative_time'], plot_data['group3_avg'], 
             color=GROUP_COLORS['group3'], label='Group 3', linewidth=2)
    if show_shadow:
        plt.fill_between(plot_data['relative_time'], 
                         plot_data['group3_avg'] - plot_data['group3_std'],
                         plot_data['group3_avg'] + plot_data['group3_std'],
                         color=GROUP_COLORS['group3'], alpha=0.2)
    
    # 添加图例和标签
    plt.legend(fontsize=12)
    plt.xlabel('Time (seconds)', fontsize=14)
    plt.ylabel('Δ F /F', fontsize=14)
    plt.title(f'Average Calcium Concentration of Neurons {n_stamps} Timestamps After CD1', fontsize=16)
    
    # 设置统一的y轴范围
    plt.ylim([-0.3, 0.5])
    
    # 添加垂直线标记CD1出现时间点
    plt.axvline(x=0, color='k', linestyle='--', linewidth=2)
    plt.text(5, 0.35, 'CD1', fontsize=14)
    
    # 网格线
    plt.grid(False)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"CD1后轨迹图已保存至: {output_path}")

def plot_combined_cd1_trace(data, cd1_index, before_stamps, after_stamps, output_path, show_shadow=True, sampling_rate=4.8):
    """
    在同一个图中绘制CD1标签前后的神经元组平均钙离子浓度轨迹图
    
    参数:
        data: 包含平均值和标准差的DataFrame
        cd1_index: CD1标签首次出现的索引
        before_stamps: CD1标签前的时间戳数量
        after_stamps: CD1标签后的时间戳数量
        output_path: 输出图像路径
        show_shadow: 是否显示标准差阴影区域
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
             color=GROUP_COLORS['group1'], label='Group 1', linewidth=2)
    if show_shadow:
        plt.fill_between(before_data['relative_time'], 
                         before_data['group1_avg'] - before_data['group1_std'],
                         before_data['group1_avg'] + before_data['group1_std'],
                         color=GROUP_COLORS['group1'], alpha=0.2)
    
    # Group 2 - 黄色
    plt.plot(before_data['relative_time'], before_data['group2_avg'], 
             color=GROUP_COLORS['group2'], label='Group 2', linewidth=2)
    if show_shadow:
        plt.fill_between(before_data['relative_time'], 
                         before_data['group2_avg'] - before_data['group2_std'],
                         before_data['group2_avg'] + before_data['group2_std'],
                         color=GROUP_COLORS['group2'], alpha=0.2)
    
    # Group 3 - 黑色
    plt.plot(before_data['relative_time'], before_data['group3_avg'], 
             color=GROUP_COLORS['group3'], label='Group 3', linewidth=2)
    if show_shadow:
        plt.fill_between(before_data['relative_time'], 
                         before_data['group3_avg'] - before_data['group3_std'],
                         before_data['group3_avg'] + before_data['group3_std'],
                         color=GROUP_COLORS['group3'], alpha=0.2)
    
    # 绘制CD1后的轨迹
    # Group 1 - 绿色
    plt.plot(after_data['relative_time'], after_data['group1_avg'], 
             color=GROUP_COLORS['group1'], linewidth=2)
    if show_shadow:
        plt.fill_between(after_data['relative_time'], 
                         after_data['group1_avg'] - after_data['group1_std'],
                         after_data['group1_avg'] + after_data['group1_std'],
                         color=GROUP_COLORS['group1'], alpha=0.2)
    
    # Group 2 - 黄色
    plt.plot(after_data['relative_time'], after_data['group2_avg'], 
             color=GROUP_COLORS['group2'], linewidth=2)
    if show_shadow:
        plt.fill_between(after_data['relative_time'], 
                         after_data['group2_avg'] - after_data['group2_std'],
                         after_data['group2_avg'] + after_data['group2_std'],
                         color=GROUP_COLORS['group2'], alpha=0.2)
    
    # Group 3 - 黑色
    plt.plot(after_data['relative_time'], after_data['group3_avg'], 
             color=GROUP_COLORS['group3'], linewidth=2)
    if show_shadow:
        plt.fill_between(after_data['relative_time'], 
                         after_data['group3_avg'] - after_data['group3_std'],
                         after_data['group3_avg'] + after_data['group3_std'],
                         color=GROUP_COLORS['group3'], alpha=0.2)
    
    # 添加图例和标签
    plt.legend(fontsize=12)
    plt.xlabel('Time (seconds)', fontsize=14)
    plt.ylabel('Δ F /F', fontsize=14)
    plt.title(f'Comparison of Average Calcium Concentration of Neurons Before and After CD1 ({before_stamps}/{after_stamps} timestamps)', fontsize=16)
    
    # 设置统一的y轴范围
    plt.ylim([-0.3, 0.5])
    
    # 添加垂直线标记CD1出现时间点
    plt.axvline(x=0, color='k', linestyle='--', linewidth=2)
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

def load_neuron_positions(file_path):
    """
    加载神经元位置坐标数据
    
    参数:
        file_path: 神经元位置坐标文件路径
        
    返回:
        positions_df: 包含神经元位置的DataFrame
    """
    print(f"正在加载神经元位置数据: {file_path}")
    
    try:
        positions_df = pd.read_csv(file_path)
        
        # 检查数据结构
        required_columns = ['number', 'relative_x', 'relative_y']
        for col in required_columns:
            if col not in positions_df.columns:
                raise ValueError(f"位置数据中缺少'{col}'列")
        
        # 神经元编号从1开始，但Python索引从0开始，所以将'number'转换为'n+数字'格式
        positions_df['neuron_id'] = 'n' + positions_df['number'].astype(int).astype(str)
        
        print(f"位置数据加载完成，共 {len(positions_df)} 个神经元")
        return positions_df
    
    except Exception as e:
        print(f"加载位置数据时出错: {str(e)}")
        return None

def plot_neuron_topology(positions_df, group1_cols, group2_cols, group3_cols, output_path):
    """
    绘制神经元空间拓扑图，按组标记不同颜色
    
    参数:
        positions_df: 包含神经元位置的DataFrame
        group1_cols, group2_cols, group3_cols: 各组神经元列名
        output_path: 输出图像路径
    """
    if positions_df is None:
        print("警告: 无法绘制神经元拓扑图，位置数据不可用")
        return
    
    # 创建图像
    plt.figure(figsize=(10, 10))
    
    # 为每个组内的神经元创建标记
    legend_elements = []
    
    # 处理第一组 - 绿色
    group1_positions = positions_df[positions_df['neuron_id'].isin(group1_cols)]
    if not group1_positions.empty:
        plt.scatter(group1_positions['relative_x'], group1_positions['relative_y'], 
                   color=GROUP_COLORS['group1'], s=200, label='Group 1')
        legend_elements.append(Patch(facecolor=GROUP_COLORS['group1'], edgecolor='black', label='Group 1'))
        
        # 添加神经元ID标签
        for _, row in group1_positions.iterrows():
            plt.text(row['relative_x'], row['relative_y'], str(int(row['number'])), 
                    fontsize=10, ha='center', va='center', color='white')
    
    # 处理第二组 - 黄色
    group2_positions = positions_df[positions_df['neuron_id'].isin(group2_cols)]
    if not group2_positions.empty:
        plt.scatter(group2_positions['relative_x'], group2_positions['relative_y'], 
                   color=GROUP_COLORS['group2'], s=200, label='Group 2')
        legend_elements.append(Patch(facecolor=GROUP_COLORS['group2'], edgecolor='black', label='Group 2'))
        
        # 添加神经元ID标签 - 黄色背景使用黑色标签更清晰
        for _, row in group2_positions.iterrows():
            plt.text(row['relative_x'], row['relative_y'], str(int(row['number'])), 
                    fontsize=10, ha='center', va='center', color='black')
    
    # 处理第三组 - 黑色
    group3_positions = positions_df[positions_df['neuron_id'].isin(group3_cols)]
    if not group3_positions.empty:
        plt.scatter(group3_positions['relative_x'], group3_positions['relative_y'], 
                   color=GROUP_COLORS['group3'], s=200, label='Group 3')
        legend_elements.append(Patch(facecolor=GROUP_COLORS['group3'], edgecolor='black', label='Group 3'))
        
        # 添加神经元ID标签
        for _, row in group3_positions.iterrows():
            plt.text(row['relative_x'], row['relative_y'], str(int(row['number'])), 
                    fontsize=10, ha='center', va='center', color='white')
    
    # 设置图表属性
    plt.title('Neuron Spatial Topology Distribution (Colored by Group)', fontsize=16)
    plt.xlabel('Relative X Coordinate', fontsize=14)
    plt.ylabel('Relative Y Coordinate', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保持坐标轴比例一致
    plt.axis('equal')
    
    # 添加图例
    plt.legend(handles=legend_elements, fontsize=12)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"神经元拓扑图已保存至: {output_path}")

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 确保输出目录存在
    ensure_dir(args.output_dir)
    
    # 加载数据
    data = load_data(args.input)
    
    # 将神经元分组
    group1_cols, group2_cols, group3_cols = group_neurons(data)
    
    # 找到CD1标签首次出现的位置
    cd1_index = find_cd1_index(data)
    
    # 计算每组神经元的平均值和标准差
    data_with_avgs = calculate_group_averages(data, group1_cols, group2_cols, group3_cols)
    
    # 生成输出文件路径
    output_before_cd1 = os.path.join(args.output_dir, "trace_before_cd1.png")
    output_after_cd1 = os.path.join(args.output_dir, "trace_after_cd1.png")
    output_combined_cd1 = os.path.join(args.output_dir, "trace_combined_cd1.png")
    output_topology = os.path.join(args.output_dir, "neuron_topology.png")
    
    # 控制是否显示阴影
    show_shadow = not args.no_shadow
    
    # 绘制CD1标签前的轨迹图
    plot_trace_before_cd1(data_with_avgs, cd1_index, args.before_stamps, output_before_cd1, show_shadow, args.sampling_rate)
    
    # 绘制CD1标签后的轨迹图
    plot_trace_after_cd1(data_with_avgs, cd1_index, args.after_stamps, output_after_cd1, show_shadow, args.sampling_rate)
    
    # 绘制CD1标签前后组合的轨迹图
    plot_combined_cd1_trace(data_with_avgs, cd1_index, args.before_stamps, args.after_stamps, output_combined_cd1, show_shadow, args.sampling_rate)
    
    # 加载神经元位置数据并绘制拓扑图
    position_data = load_neuron_positions(args.position_file)
    plot_neuron_topology(position_data, group1_cols, group2_cols, group3_cols, output_topology)
    
    print("所有图像绘制完成！")

if __name__ == "__main__":
    main()