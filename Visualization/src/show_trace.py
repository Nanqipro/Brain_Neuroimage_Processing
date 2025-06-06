import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from matplotlib.patches import Patch
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='绘制神经元分组的钙离子浓度轨迹图')
    parser.add_argument('--input', type=str, 
                        default='../datasets/No.297920240925homecagefamilarmice.xlsx',
                        help='输入数据文件路径')
    parser.add_argument('--output-dir', type=str, 
                        default='../results/CD1_traces_homecage/',
                        help='输出图像目录')
    parser.add_argument('--position-file', type=str,
                        default='../datasets/homecage_Max_position.csv',
                        help='神经元位置坐标文件路径')
    parser.add_argument('--before-stamps', type=int, default=100,
                        help='CD1标签前的时间戳数量')
    parser.add_argument('--after-stamps', type=int, default=100,
                        help='CD1标签后的时间戳数量')
    # 阴影区域参数已被移除
    parser.add_argument('--sampling-rate', type=float, default=4.8,
                        help='采样频率(Hz)，默认为4.8Hz')
    parser.add_argument('--smooth', action='store_true', default=True,
                        help='是否对曲线进行平滑处理')
    parser.add_argument('--no-smooth', action='store_false', dest='smooth',
                        help='禁用曲线平滑处理')
    parser.add_argument('--smooth-window', type=int, default=11,
                        help='平滑窗口大小（奇数），默认为11')
    parser.add_argument('--smooth-poly', type=int, default=3,
                        help='平滑多项式阶数，默认为3')
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
    # # 定义神经元分组 day3
    # group1 = ['n49', 'n32', 'n22', 'n47', 'n17', 'n12', 'n28', 'n18', 'n42', 
    #           'n46', 'n38', 'n40', 'n52', 'n7', 'n45', 'n43', 'n15', 'n16', 'n2', 'n51', 'n53']
    
    # group2 = ['n20', 'n33', 'n21', 'n5', 'n1', 'n26', 'n25', 'n36', 'n44', 
    #           'n34', 'n4', 'n24', 'n10', 'n11', 'n9', 'n3', 'n41', 'n35', 'n48', 'n19']
    
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
    
    # # 定义神经元分组 homecage
    # group1 = ['n25', 'n2', 'n40', 'n22',  'n37', 
    #           'n3', 'n24', 'n36', 'n21', 'n32', 'n26', 'n39']
    
    # group2 = ['n18', 'n14', 'n29', 'n7', 'n38', 'n10', 'n19', 'n23', 
    #           'n17', 'n11', 'n5', 'n20', 'n30', 'n33', 'n34', 'n27']



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
    
    
    
    # homecage 单独分组
    # 定义神经元分组
    group1 = ['n4', 'n41', 'n43', 'n34', 'n13', 'n33', 'n27', 'n12']
    
    group2 = ['n37', 'n30', 'n17', 'n20', 'n18', 'n32', 'n31', 'n21', 'n25', 'n11', 'n10', 'n29', 'n7', 'n22']

    
    

    
    
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
    'group3': '#808080'   # 灰色
}

def smooth_data(data, window_length=11, polyorder=3):
    """
    使用Savitzky-Golay滤波器平滑数据
    
    Parameters
    ----------
    data : array-like
        需要平滑的数据
    window_length : int
        滤波器窗口长度（必须为奇数）
    polyorder : int
        多项式拟合阶数
        
    Returns
    -------
    smoothed_data : array-like
        平滑后的数据
    """
    if len(data) < window_length:
        # 如果数据长度小于窗口长度，则调整窗口大小
        window_length = len(data) if len(data) % 2 == 1 else len(data) - 1
        if window_length < 3:
            return data  # 数据太短，无法平滑
    
    try:
        smoothed = savgol_filter(data, window_length=window_length, polyorder=polyorder)
        return smoothed
    except Exception as e:
        print(f"平滑处理出错: {e}，返回原始数据")
        return data

def interpolate_data(x_data, y_data, factor=3):
    """
    使用插值方法增加数据点数量，使曲线更平滑
    
    Parameters
    ----------
    x_data : array-like
        X轴数据
    y_data : array-like
        Y轴数据
    factor : int
        插值倍数，默认为3倍
        
    Returns
    -------
    x_interp : array-like
        插值后的X轴数据
    y_interp : array-like
        插值后的Y轴数据
    """
    try:
        # 创建插值函数
        f_interp = interp1d(x_data, y_data, kind='cubic', bounds_error=False, fill_value='extrapolate')
        
        # 生成更密集的X轴点
        x_interp = np.linspace(x_data.min(), x_data.max(), len(x_data) * factor)
        y_interp = f_interp(x_interp)
        
        return x_interp, y_interp
    except Exception as e:
        print(f"插值处理出错: {e}，返回原始数据")
        return x_data, y_data

def plot_trace_before_cd1(data, cd1_index, n_stamps, output_path, sampling_rate=4.8, 
                          enable_smooth=True, smooth_window=11, smooth_poly=3):
    """
    绘制CD1标签前n_stamps个时间戳的神经元组平均钙离子浓度轨迹图
    
    Parameters
    ----------
    data : DataFrame
        包含平均值和标准差的DataFrame
    cd1_index : int
        CD1标签首次出现的索引
    n_stamps : int
        要绘制的时间戳数量
    output_path : str
        输出图像路径
    sampling_rate : float
        采样频率(Hz)
    enable_smooth : bool
        是否启用曲线平滑
    smooth_window : int
        平滑窗口大小
    smooth_poly : int
        平滑多项式阶数
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
    
    # 准备绘图数据
    time_data = plot_data['relative_time'].values
    group1_data = plot_data['group1_avg'].values
    group2_data = plot_data['group2_avg'].values
    group3_data = plot_data['group3_avg'].values
    
    # 应用平滑处理
    if enable_smooth:
        print(f"正在应用平滑处理（窗口大小: {smooth_window}, 多项式阶数: {smooth_poly}）...")
        group1_data = smooth_data(group1_data, window_length=smooth_window, polyorder=smooth_poly)
        group2_data = smooth_data(group2_data, window_length=smooth_window, polyorder=smooth_poly)
        group3_data = smooth_data(group3_data, window_length=smooth_window, polyorder=smooth_poly)
        
        # 可选：应用插值增加数据点密度
        time_data, group1_data = interpolate_data(time_data, group1_data, factor=2)
        _, group2_data = interpolate_data(plot_data['relative_time'].values, group2_data, factor=2)
        _, group3_data = interpolate_data(plot_data['relative_time'].values, group3_data, factor=2)
    
    # 创建图像
    plt.figure(figsize=(12, 8))
    
    # 使用统一的颜色方案绘制三条曲线及其阴影区域
    # Group 1 - 绿色
    plt.plot(time_data, group1_data, 
             color=GROUP_COLORS['group1'], label='Group 1', linewidth=3)
    # 阴影区域显示已移除
    
    # Group 2 - 黄色
    plt.plot(time_data, group2_data, 
             color=GROUP_COLORS['group2'], label='Group 2', linewidth=3)
    # 阴影区域显示已移除
    
    # Group 3 - 灰色虚线
    plt.plot(time_data, group3_data, 
             color=GROUP_COLORS['group3'], label='Group 3', linewidth=3, linestyle='--')
    # 阴影区域显示已移除
    
    # 添加图例和标签
    plt.legend(fontsize=12)
    plt.xlabel('Time (seconds)', fontsize=14)
    plt.ylabel('Δ F /F', fontsize=14)
    plt.title(f'Average Calcium Concentration of Neurons {n_stamps} Timestamps Before CD1', fontsize=16)
    
    # 设置统一的y轴范围
    plt.ylim([-0.6, 0.5])
    
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

def plot_trace_after_cd1(data, cd1_index, n_stamps, output_path, sampling_rate=4.8,
                         enable_smooth=True, smooth_window=11, smooth_poly=3):
    """
    绘制CD1标签后n_stamps个时间戳的神经元组平均钙离子浓度轨迹图
    
    Parameters
    ----------
    data : DataFrame
        包含平均值和标准差的DataFrame
    cd1_index : int
        CD1标签首次出现的索引
    n_stamps : int
        要绘制的时间戳数量
    output_path : str
        输出图像路径
    sampling_rate : float
        采样频率(Hz)
    enable_smooth : bool
        是否启用曲线平滑
    smooth_window : int
        平滑窗口大小
    smooth_poly : int
        平滑多项式阶数
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
    
    # 准备绘图数据
    time_data = plot_data['relative_time'].values
    group1_data = plot_data['group1_avg'].values
    group2_data = plot_data['group2_avg'].values
    group3_data = plot_data['group3_avg'].values
    
    # 应用平滑处理
    if enable_smooth:
        print(f"正在应用平滑处理（窗口大小: {smooth_window}, 多项式阶数: {smooth_poly}）...")
        group1_data = smooth_data(group1_data, window_length=smooth_window, polyorder=smooth_poly)
        group2_data = smooth_data(group2_data, window_length=smooth_window, polyorder=smooth_poly)
        group3_data = smooth_data(group3_data, window_length=smooth_window, polyorder=smooth_poly)
        
        # 可选：应用插值增加数据点密度
        time_data, group1_data = interpolate_data(time_data, group1_data, factor=2)
        _, group2_data = interpolate_data(plot_data['relative_time'].values, group2_data, factor=2)
        _, group3_data = interpolate_data(plot_data['relative_time'].values, group3_data, factor=2)
    
    # 创建图像
    plt.figure(figsize=(12, 8))
    
    # 使用统一的颜色方案绘制三条曲线及其阴影区域
    # Group 1 - 绿色
    plt.plot(time_data, group1_data, 
             color=GROUP_COLORS['group1'], label='Group 1', linewidth=3)
    # 阴影区域显示已移除
    
    # Group 2 - 黄色
    plt.plot(time_data, group2_data, 
             color=GROUP_COLORS['group2'], label='Group 2', linewidth=3)
    # 阴影区域显示已移除
    
    # Group 3 - 灰色虚线
    plt.plot(time_data, group3_data, 
             color=GROUP_COLORS['group3'], label='Group 3', linewidth=3, linestyle='--')
    # 阴影区域显示已移除
    
    # 添加图例和标签
    plt.legend(fontsize=12)
    plt.xlabel('Time (seconds)', fontsize=14)
    plt.ylabel('Δ F /F', fontsize=14)
    plt.title(f'Average Calcium Concentration of Neurons {n_stamps} Timestamps After CD1', fontsize=16)
    
    # 设置统一的y轴范围
    plt.ylim([-0.6, 0.5])
    
    # 添加垂直线标记CD1出现时间点
    plt.axvline(x=0, color='k', linestyle='--', linewidth=3)
    plt.text(5, 0.35, 'CD1', fontsize=14)
    
    # 网格线
    plt.grid(False)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"CD1后轨迹图已保存至: {output_path}")

def plot_combined_cd1_trace(data, cd1_index, before_stamps, after_stamps, output_path, sampling_rate=4.8,
                           enable_smooth=True, smooth_window=11, smooth_poly=3):
    """
    在同一个图中绘制CD1标签前后的神经元组平均钙离子浓度轨迹图
    
    Parameters
    ----------
    data : DataFrame
        包含平均值和标准差的DataFrame
    cd1_index : int
        CD1标签首次出现的索引
    before_stamps : int
        CD1标签前的时间戳数量
    after_stamps : int
        CD1标签后的时间戳数量
    output_path : str
        输出图像路径
    sampling_rate : float
        采样频率(Hz)
    enable_smooth : bool
        是否启用曲线平滑
    smooth_window : int
        平滑窗口大小
    smooth_poly : int
        平滑多项式阶数
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
    
    # 准备绘图数据
    before_time = before_data['relative_time'].values
    before_group1 = before_data['group1_avg'].values
    before_group2 = before_data['group2_avg'].values
    before_group3 = before_data['group3_avg'].values
    
    after_time = after_data['relative_time'].values
    after_group1 = after_data['group1_avg'].values
    after_group2 = after_data['group2_avg'].values
    after_group3 = after_data['group3_avg'].values
    
    # 应用平滑处理
    if enable_smooth:
        print(f"正在应用平滑处理（窗口大小: {smooth_window}, 多项式阶数: {smooth_poly}）...")
        # 平滑CD1前的数据
        before_group1 = smooth_data(before_group1, window_length=smooth_window, polyorder=smooth_poly)
        before_group2 = smooth_data(before_group2, window_length=smooth_window, polyorder=smooth_poly)
        before_group3 = smooth_data(before_group3, window_length=smooth_window, polyorder=smooth_poly)
        
        # 平滑CD1后的数据
        after_group1 = smooth_data(after_group1, window_length=smooth_window, polyorder=smooth_poly)
        after_group2 = smooth_data(after_group2, window_length=smooth_window, polyorder=smooth_poly)
        after_group3 = smooth_data(after_group3, window_length=smooth_window, polyorder=smooth_poly)
        
        # 可选：应用插值增加数据点密度
        before_time, before_group1 = interpolate_data(before_time, before_group1, factor=2)
        _, before_group2 = interpolate_data(before_data['relative_time'].values, before_group2, factor=2)
        _, before_group3 = interpolate_data(before_data['relative_time'].values, before_group3, factor=2)
        
        after_time, after_group1 = interpolate_data(after_time, after_group1, factor=2)
        _, after_group2 = interpolate_data(after_data['relative_time'].values, after_group2, factor=2)
        _, after_group3 = interpolate_data(after_data['relative_time'].values, after_group3, factor=2)
    
    # 创建图像，增加图表大小以解决布局问题
    plt.figure(figsize=(15, 10))
    
    # 绘制CD1前的轨迹
    # Group 1 - 绿色
    plt.plot(before_time, before_group1, 
             color=GROUP_COLORS['group1'], label='Group 1', linewidth=3)
    # 阴影区域显示已移除
    
    # Group 2 - 黄色
    plt.plot(before_time, before_group2, 
             color=GROUP_COLORS['group2'], label='Group 2', linewidth=3)
    # 阴影区域显示已移除
    
    # Group 3 - 灰色虚线
    plt.plot(before_time, before_group3, 
             color=GROUP_COLORS['group3'], label='Group 3', linewidth=3, linestyle='--')
    # 阴影区域显示已移除
    
    # 绘制CD1后的轨迹
    # Group 1 - 绿色
    plt.plot(after_time, after_group1, 
             color=GROUP_COLORS['group1'], linewidth=3)
    # 阴影区域显示已移除
    
    # Group 2 - 黄色
    plt.plot(after_time, after_group2, 
             color=GROUP_COLORS['group2'], linewidth=3)
    # 阴影区域显示已移除
    
    # Group 3 - 灰色虚线
    plt.plot(after_time, after_group3, 
             color=GROUP_COLORS['group3'], linewidth=3, linestyle='--')
    # 阴影区域显示已移除
    
    # 添加图例和标签
    plt.legend(fontsize=12)
    plt.xlabel('Time (seconds)', fontsize=14)
    plt.ylabel('Δ F /F', fontsize=14)
    plt.title(f'Comparison of Average Calcium Concentration of Neurons Before and After CD1 ({before_stamps}/{after_stamps} timestamps)', fontsize=16)
    
    # 设置统一的y轴范围
    plt.ylim([-0.6, 0.5])
    
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
    
    # 阴影区域控制已被移除
    
    # 绘制CD1标签前的轨迹图
    plot_trace_before_cd1(data_with_avgs, cd1_index, args.before_stamps, output_before_cd1, 
                         sampling_rate=args.sampling_rate, enable_smooth=args.smooth,
                         smooth_window=args.smooth_window, smooth_poly=args.smooth_poly)
    
    # 绘制CD1标签后的轨迹图
    plot_trace_after_cd1(data_with_avgs, cd1_index, args.after_stamps, output_after_cd1, 
                        sampling_rate=args.sampling_rate, enable_smooth=args.smooth,
                        smooth_window=args.smooth_window, smooth_poly=args.smooth_poly)
    
    # 绘制CD1标签前后组合的轨迹图
    plot_combined_cd1_trace(data_with_avgs, cd1_index, args.before_stamps, args.after_stamps, output_combined_cd1, 
                           sampling_rate=args.sampling_rate, enable_smooth=args.smooth,
                           smooth_window=args.smooth_window, smooth_poly=args.smooth_poly)
    
    # 加载神经元位置数据并绘制拓扑图
    position_data = load_neuron_positions(args.position_file)
    plot_neuron_topology(position_data, group1_cols, group2_cols, group3_cols, output_topology)
    
    print("所有图像绘制完成！")

if __name__ == "__main__":
    main()