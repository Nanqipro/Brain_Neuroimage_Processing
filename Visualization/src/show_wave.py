#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
神经元钙离子波动可视化模块

该模块用于将每条神经元的钙离子波动数据可视化展现为图表，
横坐标为时间戳（stamp），纵坐标为钙离子浓度。
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import seaborn as sns
from typing import Union, List, Dict, Optional, Tuple

def load_data(file_path: str) -> pd.DataFrame:
    """
    加载神经元钙离子数据
    
    Parameters
    ----------
    file_path : str
        数据文件路径，支持.md, .csv, .xlsx格式
    
    Returns
    -------
    pd.DataFrame
        包含神经元钙离子数据的DataFrame
    """
    if file_path.endswith('.md'):
        # 尝试从markdown文件中解析表格数据
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # 提取markdown表格内容
            lines = content.strip().split('\n')
            # 跳过表头和分隔线
            data_lines = [line.strip().strip('|').split('|') for line in lines[2:]]
            columns = [col.strip() for col in lines[0].strip().strip('|').split('|')]
            
            # 将数据转换为DataFrame
            df = pd.DataFrame(data_lines, columns=columns)
            # 将数值列转换为浮点数
            for col in df.columns:
                if col != 'behavior':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
    elif file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {file_path}")

def plot_neuron_calcium(df: pd.DataFrame, neuron_id: str, save_dir: str = None) -> Figure:
    """
    绘制单个神经元的钙离子波动图
    
    Parameters
    ----------
    df : pd.DataFrame
        包含神经元数据的DataFrame
    neuron_id : str
        神经元ID，对应DataFrame中的列名
    save_dir : str, optional
        图像保存目录，默认为None（不保存）
    
    Returns
    -------
    Figure  
        matplotlib图像对象
    """
    if neuron_id not in df.columns:
        raise ValueError(f"神经元ID {neuron_id} 在数据中不存在")
    
    # 创建图像
    fig, ax = plt.subplots(figsize=(60, 10))
    
    # 绘制钙离子浓度变化曲线
    ax.plot(df['stamp'], df[neuron_id], linewidth=2, color='#1f77b4')
    
    # 添加平滑曲线（移动平均）
    window_size = min(10, len(df) // 10)  # 根据数据长度调整窗口大小
    if window_size > 1:
        smooth_data = df[neuron_id].rolling(window=window_size, center=True).mean()
        ax.plot(df['stamp'], smooth_data, linewidth=2.5, color='#ff7f0e', alpha=0.7, 
                label=f'Smooth Curve (Window={window_size})')
    
    # 设置图表样式和标题
    ax.set_title(f'Neuron {neuron_id} Calcium Concentration Fluctuation', fontsize=16)
    ax.set_xlabel('Timestamp (stamp)', fontsize=14)
    ax.set_ylabel('Calcium Concentration', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 设置横坐标刻度，每50个stamp显示一个刻度
    stamps = df['stamp'].values
    min_stamp = stamps.min()
    max_stamp = stamps.max()
    step = 50  # 每50个stamp显示一个刻度
    
    # 计算刻度位置，从最小值开始，以step为间隔
    tick_positions = np.arange(min_stamp, max_stamp + step, step)
    
    # 设置横坐标刻度
    ax.set_xticks(tick_positions)
    ax.tick_params(axis='x', rotation=45)  # 旋转刻度标签以防重叠
    
    # 突出显示极值点
    max_val = df[neuron_id].max()
    max_idx = df[neuron_id].idxmax()
    min_val = df[neuron_id].min()
    min_idx = df[neuron_id].idxmin()
    
    ax.scatter(df.loc[max_idx, 'stamp'], max_val, color='red', s=100, zorder=5, 
               label=f'Maximum: {max_val:.2f}')
    ax.scatter(df.loc[min_idx, 'stamp'], min_val, color='green', s=100, zorder=5, 
               label=f'Minimum: {min_val:.2f}')
    
    # 添加统计信息
    mean_val = df[neuron_id].mean()
    std_val = df[neuron_id].std()
    stats_text = f'Mean: {mean_val:.2f}\nStandard Deviation: {std_val:.2f}'
    ax.text(0.02, 0.97, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
    
    # 添加图例
    ax.legend(loc='best')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像（如果指定了保存路径）
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'neuron_{neuron_id}_calcium.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存至: {save_path}")
    
    return fig

def plot_all_neurons(df: pd.DataFrame, save_dir: str = None, 
                     exclude_cols: List[str] = ['stamp', 'behavior']) -> List[Figure]:
    """
    绘制所有神经元的钙离子波动图
    
    Parameters
    ----------
    df : pd.DataFrame
        包含神经元数据的DataFrame
    save_dir : str, optional
        图像保存目录，默认为None（由调用函数指定）
    exclude_cols : List[str], optional
        需要排除的列名列表，默认排除'stamp'和'behavior'列
    
    Returns
    -------
    List[Figure]
        所有图像对象的列表
    """
    # 确保保存目录存在
    if save_dir is None:
        save_dir = '../results/calcium_waves'
    os.makedirs(save_dir, exist_ok=True)
    
    # 使用element_extraction.py中的方法获取所有神经元列
    neuron_cols = [col for col in df.columns if col.startswith('n') and col[1:].isdigit()]
    print(f"检测到 {len(neuron_cols)} 个神经元数据列")
    
    # 绘制并保存每个神经元的图像
    figures = []
    for neuron_id in neuron_cols:
        try:
            fig = plot_neuron_calcium(df, neuron_id, save_dir)
            figures.append(fig)
            plt.close(fig)  # 关闭图像以释放内存
        except Exception as e:
            print(f"处理神经元 {neuron_id} 时出错: {e}")
    
    print(f"已为 {len(figures)} 个神经元生成钙离子波动图，保存至 {save_dir}")
    return figures

def generate_summary_report(df: pd.DataFrame, save_path: str = None) -> None:
    """
    生成神经元钙离子数据的摘要报告
    
    Parameters
    ----------
    df : pd.DataFrame
        包含神经元数据的DataFrame
    save_path : str, optional
        报告保存路径，默认为None（由调用函数指定）
    """
    # 确保目录存在
    if save_path is None:
        save_path = '../results/neuron_summary.html'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 使用element_extraction.py中的方法获取所有神经元列
    neuron_cols = [col for col in df.columns if col.startswith('n') and col[1:].isdigit()]
    
    # 计算每个神经元的统计量
    stats = []
    for col in neuron_cols:
        stats.append({
            '神经元ID': col,
            '最大值': df[col].max(),
            '最小值': df[col].min(),
            '均值': df[col].mean(),
            '标准差': df[col].std(),
            '变异系数': df[col].std() / df[col].mean() if df[col].mean() != 0 else np.nan,
            '波动范围': df[col].max() - df[col].min()
        })
    
    # 创建统计摘要DataFrame
    stats_df = pd.DataFrame(stats)
    
    # 生成HTML报告
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('<html><head><title>神经元钙离子波动分析报告</title>')
        f.write('<style>body{font-family:Arial;margin:20px;} table{border-collapse:collapse;width:100%;}')
        f.write('th,td{border:1px solid #ddd;padding:8px;text-align:left;}')
        f.write('th{background-color:#f2f2f2;} tr:nth-child(even){background-color:#f9f9f9;}')
        f.write('</style></head><body>')
        f.write('<h1>神经元钙离子波动分析报告</h1>')
        f.write(f'<p>分析时间: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}</p>')
        f.write(f'<p>数据样本数: {len(df)}</p>')
        f.write(f'<p>神经元数量: {len(neuron_cols)}</p>')
        f.write('<h2>神经元统计摘要</h2>')
        f.write(stats_df.to_html(index=False, float_format='%.4f'))
        f.write('</body></html>')
    
    print(f"分析报告已保存至: {save_path}")

def main():
    """
    主函数，用于处理命令行参数并执行可视化
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='神经元钙离子波动可视化工具')
    parser.add_argument('--data', type=str, default='../datasets/processed_Day6.xlsx',
                        help='数据文件路径，支持.md, .csv, .xlsx格式')
    parser.add_argument('--output', type=str, default=None,
                        help='图像保存目录，不指定则根据数据集名称自动生成')
    parser.add_argument('--neuron', type=str, default=None,
                        help='指定要可视化的神经元ID，不指定则处理所有神经元')
    parser.add_argument('--report', action='store_true',
                        help='生成神经元统计摘要报告')
    
    args = parser.parse_args()
    
    # 加载数据
    print(f"正在加载数据文件: {args.data}")
    df = load_data(args.data)
    print(f"数据加载完成，共有 {len(df)} 条记录和 {len(df.columns)} 个列")
    
    # 根据数据文件名生成输出目录
    if args.output is None:
        # 提取数据文件名（不含扩展名）
        data_basename = os.path.basename(args.data)
        dataset_name = os.path.splitext(data_basename)[0]
        output_dir = f"../results/{dataset_name}/calcium_waves"
    else:
        output_dir = args.output
    
    print(f"输出目录设置为: {output_dir}")
    
    # 根据参数执行可视化
    if args.neuron:
        print(f"正在为神经元 {args.neuron} 生成图像...")
        plot_neuron_calcium(df, args.neuron, output_dir)
    else:
        print("正在为所有神经元生成图像...")
        plot_all_neurons(df, output_dir)
    
    # 生成报告（如果需要）
    if args.report:
        print("正在生成统计摘要报告...")
        report_dir = os.path.dirname(output_dir)
        report_path = os.path.join(report_dir, 'neuron_summary.html')
        generate_summary_report(df, report_path)
    
    print("处理完成!")

if __name__ == "__main__":
    main()
