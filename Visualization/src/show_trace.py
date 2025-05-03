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
                        default='../results/CD1_traces/',
                        help='输出图像目录')
    parser.add_argument('--before-stamps', type=int, default=50,
                        help='CD1标签前的时间戳数量')
    parser.add_argument('--after-stamps', type=int, default=50,
                        help='CD1标签后的时间戳数量')
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
    # 定义神经元分组
    group1 = ['n49', 'n32', 'n22', 'n47', 'n17', 'n12', 'n28', 'n18', 'n42', 
              'n46', 'n38', 'n40', 'n52', 'n7', 'n45', 'n43']
    
    group2 = ['n20', 'n33', 'n21', 'n5', 'n1', 'n26', 'n25', 'n36', 'n44', 
              'n34', 'n4', 'n24', 'n10', 'n11']
    
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
    'group1': '#E41A1C',  # 红色
    'group2': '#377EB8',  # 蓝色
    'group3': '#4DAF4A'   # 绿色
}

def plot_trace_before_cd1(data, cd1_index, n_stamps, output_path):
    """
    绘制CD1标签前n_stamps个时间戳的神经元组平均钙离子浓度轨迹图
    
    参数:
        data: 包含平均值和标准差的DataFrame
        cd1_index: CD1标签首次出现的索引
        n_stamps: 要绘制的时间戳数量
        output_path: 输出图像路径
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
    
    # 创建相对时间轴（以秒为单位，假设每个时间戳间隔为0.1秒）
    # 注意：这里调整为从0开始，到CD1出现前的时间
    plot_data['relative_time'] = [(i * 0.1) for i in range(len(plot_data))]
    
    # 创建图像
    plt.figure(figsize=(12, 8))
    
    # 使用统一的颜色方案绘制三条曲线及其阴影区域
    # 组1
    plt.plot(plot_data['relative_time'], plot_data['group1_avg'], 
             color=GROUP_COLORS['group1'], label='组1', linewidth=2)
    plt.fill_between(plot_data['relative_time'], 
                     plot_data['group1_avg'] - plot_data['group1_std'],
                     plot_data['group1_avg'] + plot_data['group1_std'],
                     color=GROUP_COLORS['group1'], alpha=0.3)
    
    # 组2
    plt.plot(plot_data['relative_time'], plot_data['group2_avg'], 
             color=GROUP_COLORS['group2'], label='组2', linewidth=2)
    plt.fill_between(plot_data['relative_time'], 
                     plot_data['group2_avg'] - plot_data['group2_std'],
                     plot_data['group2_avg'] + plot_data['group2_std'],
                     color=GROUP_COLORS['group2'], alpha=0.3)
    
    # 组3
    plt.plot(plot_data['relative_time'], plot_data['group3_avg'], 
             color=GROUP_COLORS['group3'], label='组3', linewidth=2)
    plt.fill_between(plot_data['relative_time'], 
                     plot_data['group3_avg'] - plot_data['group3_std'],
                     plot_data['group3_avg'] + plot_data['group3_std'],
                     color=GROUP_COLORS['group3'], alpha=0.3)
    
    # 添加图例和标签
    plt.legend(fontsize=12)
    plt.xlabel('时间 (秒)', fontsize=14)
    plt.ylabel('平均钙离子浓度', fontsize=14)
    plt.title(f'CD1标签前{n_stamps}个时间戳的神经元平均钙离子浓度', fontsize=16)
    
    # 添加垂直线标记CD1出现时间点
    plt.axvline(x=plot_data['relative_time'].max(), color='k', linestyle='--', linewidth=2)
    plt.text(plot_data['relative_time'].max()-5, plt.ylim()[1]*0.9, 'CD1', fontsize=14)
    
    # 网格线
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"CD1前轨迹图已保存至: {output_path}")

def plot_trace_after_cd1(data, cd1_index, n_stamps, output_path):
    """
    绘制CD1标签后n_stamps个时间戳的神经元组平均钙离子浓度轨迹图
    
    参数:
        data: 包含平均值和标准差的DataFrame
        cd1_index: CD1标签首次出现的索引
        n_stamps: 要绘制的时间戳数量
        output_path: 输出图像路径
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
    
    # 创建相对时间轴（以秒为单位，假设每个时间戳间隔为0.1秒）
    # 注意：这里调整为从0开始，表示从CD1出现开始的时间
    plot_data['relative_time'] = [(i * 0.1) for i in range(len(plot_data))]
    
    # 创建图像
    plt.figure(figsize=(12, 8))
    
    # 使用统一的颜色方案绘制三条曲线及其阴影区域
    # 组1
    plt.plot(plot_data['relative_time'], plot_data['group1_avg'], 
             color=GROUP_COLORS['group1'], label='组1', linewidth=2)
    plt.fill_between(plot_data['relative_time'], 
                     plot_data['group1_avg'] - plot_data['group1_std'],
                     plot_data['group1_avg'] + plot_data['group1_std'],
                     color=GROUP_COLORS['group1'], alpha=0.3)
    
    # 组2
    plt.plot(plot_data['relative_time'], plot_data['group2_avg'], 
             color=GROUP_COLORS['group2'], label='组2', linewidth=2)
    plt.fill_between(plot_data['relative_time'], 
                     plot_data['group2_avg'] - plot_data['group2_std'],
                     plot_data['group2_avg'] + plot_data['group2_std'],
                     color=GROUP_COLORS['group2'], alpha=0.3)
    
    # 组3
    plt.plot(plot_data['relative_time'], plot_data['group3_avg'], 
             color=GROUP_COLORS['group3'], label='组3', linewidth=2)
    plt.fill_between(plot_data['relative_time'], 
                     plot_data['group3_avg'] - plot_data['group3_std'],
                     plot_data['group3_avg'] + plot_data['group3_std'],
                     color=GROUP_COLORS['group3'], alpha=0.3)
    
    # 添加图例和标签
    plt.legend(fontsize=12)
    plt.xlabel('时间 (秒)', fontsize=14)
    plt.ylabel('平均钙离子浓度', fontsize=14)
    plt.title(f'CD1标签后{n_stamps}个时间戳的神经元平均钙离子浓度', fontsize=16)
    
    # 添加垂直线标记CD1出现时间点
    plt.axvline(x=0, color='k', linestyle='--', linewidth=2)
    plt.text(5, plt.ylim()[1]*0.9, 'CD1', fontsize=14)
    
    # 网格线
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"CD1后轨迹图已保存至: {output_path}")

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
    
    # 绘制CD1标签前的轨迹图
    plot_trace_before_cd1(data_with_avgs, cd1_index, args.before_stamps, output_before_cd1)
    
    # 绘制CD1标签后的轨迹图
    plot_trace_after_cd1(data_with_avgs, cd1_index, args.after_stamps, output_after_cd1)
    
    print("轨迹图绘制完成！")

if __name__ == "__main__":
    main()