#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从CSV文件读取训练历史并生成可视化图表

使用方法：
    # 从单个CSV文件生成图表
    python plot_from_csv.py --csv path/to/training_history.csv
    
    # 从多个CSV文件对比绘图
    python plot_from_csv.py --csv file1.csv file2.csv file3.csv --labels Model1 Model2 Model3
    
    # 指定输出目录
    python plot_from_csv.py --csv path/to/training_history.csv --output_dir my_plots
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import json
import glob
import re
from pathlib import Path
from collections import defaultdict

# 设置matplotlib样式
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams['font.family'] = 'Arial'
plt.style.use('seaborn-v0_8-white')


def plot_cumulative_time(csv_files, labels=None, output_dir='plots'):
    """绘制累积时间曲线
    
    Args:
        csv_files: CSV文件路径列表
        labels: 图例标签列表
        output_dir: 输出目录
    """
    plt.figure(figsize=(12, 6))
    
    if labels is None:
        labels = [f'Run {i+1}' for i in range(len(csv_files))]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(csv_files)))
    
    for idx, (csv_file, label) in enumerate(zip(csv_files, labels)):
        df = pd.read_csv(csv_file)
        plt.plot(df['epoch'], df['cumulative_time'], 
                marker='o', linewidth=2, markersize=4, 
                label=label, color=colors[idx])
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Cumulative Time (seconds)', fontsize=12)
    plt.title('Cumulative Training Time from Start', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    output_path = f'{output_dir}/cumulative_time.png'
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ 累积时间图已保存: {output_path}")


def plot_epoch_time(csv_files, labels=None, output_dir='plots'):
    """绘制每个epoch的单独执行时间
    
    Args:
        csv_files: CSV文件路径列表
        labels: 图例标签列表
        output_dir: 输出目录
    """
    plt.figure(figsize=(12, 6))
    
    if labels is None:
        labels = [f'Run {i+1}' for i in range(len(csv_files))]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(csv_files)))
    
    for idx, (csv_file, label) in enumerate(zip(csv_files, labels)):
        df = pd.read_csv(csv_file)
        plt.plot(df['epoch'], df['epoch_time'], 
                marker='o', linewidth=2, markersize=4, 
                label=label, color=colors[idx])
        
        # 添加平均线
        avg_time = df['epoch_time'].mean()
        plt.axhline(y=avg_time, color=colors[idx], linestyle='--', 
                   linewidth=1, alpha=0.5, 
                   label=f'{label} Avg: {avg_time:.3f}s')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Time per Epoch (seconds)', fontsize=12)
    plt.title('Training Time per Epoch', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, ncol=2)
    plt.tight_layout()
    
    output_path = f'{output_dir}/epoch_time.png'
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Epoch时间图已保存: {output_path}")


def plot_training_metrics(csv_files, labels=None, output_dir='plots'):
    """绘制训练和验证指标
    
    Args:
        csv_files: CSV文件路径列表
        labels: 图例标签列表
        output_dir: 输出目录
    """
    if labels is None:
        labels = [f'Run {i+1}' for i in range(len(csv_files))]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(csv_files)))
    
    # 创建2x2子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    metrics = [
        ('train_loss', 'val_f1', 'Loss & F1 Score'),
        ('train_f1', 'val_f1', 'F1 Score'),
        ('train_accuracy', 'val_accuracy', 'Accuracy'),
        ('train_precision', 'val_precision', 'Precision')
    ]
    
    for idx, (csv_file, label) in enumerate(zip(csv_files, labels)):
        df = pd.read_csv(csv_file)
        
        # Loss & F1
        ax = axes[0, 0]
        ax2 = ax.twinx()
        line1 = ax.plot(df['epoch'], df['train_loss'], 
                       linestyle='-', linewidth=2, color=colors[idx],
                       label=f'{label} Train Loss')
        line2 = ax2.plot(df['epoch'], df['val_f1'], 
                        linestyle='--', linewidth=2, color=colors[idx],
                        label=f'{label} Val F1')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss', color='b')
        ax2.set_ylabel('F1 Score', color='r')
        ax.grid(True, alpha=0.3)
        ax.set_title('Training Loss & Validation F1')
        
        # F1 Score
        ax = axes[0, 1]
        ax.plot(df['epoch'], df['train_f1'], 
               linestyle='-', linewidth=2, color=colors[idx],
               label=f'{label} Train')
        ax.plot(df['epoch'], df['val_f1'], 
               linestyle='--', linewidth=2, color=colors[idx],
               label=f'{label} Val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('F1 Score')
        ax.grid(True, alpha=0.3)
        ax.set_title('F1 Score')
        ax.legend(fontsize=9)
        
        # Accuracy
        ax = axes[1, 0]
        ax.plot(df['epoch'], df['train_accuracy'], 
               linestyle='-', linewidth=2, color=colors[idx],
               label=f'{label} Train')
        ax.plot(df['epoch'], df['val_accuracy'], 
               linestyle='--', linewidth=2, color=colors[idx],
               label=f'{label} Val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.grid(True, alpha=0.3)
        ax.set_title('Accuracy')
        ax.legend(fontsize=9)
        
        # Precision
        ax = axes[1, 1]
        ax.plot(df['epoch'], df['train_precision'], 
               linestyle='-', linewidth=2, color=colors[idx],
               label=f'{label} Train')
        ax.plot(df['epoch'], df['val_precision'], 
               linestyle='--', linewidth=2, color=colors[idx],
               label=f'{label} Val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Precision')
        ax.grid(True, alpha=0.3)
        ax.set_title('Precision')
        ax.legend(fontsize=9)
    
    plt.tight_layout()
    output_path = f'{output_dir}/training_metrics.png'
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ 训练指标图已保存: {output_path}")


def print_statistics(csv_files, labels=None):
    """打印统计信息
    
    Args:
        csv_files: CSV文件路径列表
        labels: 图例标签列表
    """
    if labels is None:
        labels = [f'Run {i+1}' for i in range(len(csv_files))]
    
    print("\n" + "=" * 80)
    print("📊 训练历史统计")
    print("=" * 80)
    
    for csv_file, label in zip(csv_files, labels):
        df = pd.read_csv(csv_file)
        
        print(f"\n【{label}】")
        print(f"  文件: {csv_file}")
        print(f"  总Epoch数: {len(df)}")
        print(f"  累积总时间: {df['cumulative_time'].iloc[-1]:.3f}秒")
        print(f"  平均Epoch时间: {df['epoch_time'].mean():.3f} ± {df['epoch_time'].std():.3f}秒")
        print(f"  最快Epoch: {df['epoch_time'].min():.3f}秒")
        print(f"  最慢Epoch: {df['epoch_time'].max():.3f}秒")
        print(f"  最终训练F1: {df['train_f1'].iloc[-1]:.4f}")
        print(f"  最佳验证F1: {df['val_f1'].max():.4f} (Epoch {df['val_f1'].idxmax() + 1})")
        print(f"  最终验证F1: {df['val_f1'].iloc[-1]:.4f}")
    
    print("=" * 80 + "\n")


def collect_scaling_data(result_base_dir='result'):
    """
    从result目录收集不同数据集规模和模型的时间统计数据
    
    Args:
        result_base_dir: 结果目录基础路径
        
    Returns:
        dict: 包含各个模型在不同规模下的时间数据
              格式: {model_name: {graph_size: {'total_time': [], 'avg_time': [], 'time_per_graph': [], 'time_per_graph_per_epoch': []}}}
    """
    data = defaultdict(lambda: defaultdict(lambda: {
        'total_time': [], 
        'avg_time': [], 
        'time_per_graph': [], 
        'time_per_graph_ms': [],
        'time_per_graph_per_epoch': [],
        'time_per_graph_per_epoch_ms': [],
        'peak_cpu_memory_mb': [],
        'avg_cpu_memory_mb': [],
        'peak_gpu_memory_mb': [],
        'avg_gpu_memory_mb': [],
        'num_edges': None,  # 边数量（对于同一规模数据集应该相同）
        'oom_error': False,  # 是否发生OOM错误
        'oom_count': 0  # OOM错误的次数
    }))
    
    # 遍历所有experiment_results.json文件（优先，包含完整信息）
    result_path = Path(result_base_dir)
    json_files = result_path.rglob('experiment_results.json')
    
    for json_file in json_files:
        try:
            # 从路径中提取信息: result/custom/graphs_XXX/MODEL/...
            path_parts = json_file.parts
            
            # 找到graphs_XXX部分
            graph_size = None
            model_name = None
            for i, part in enumerate(path_parts):
                if part.startswith('graphs_'):
                    match = re.match(r'graphs_(\d+)', part)
                    if match:
                        graph_size = int(match.group(1))
                        # 下一个部分应该是模型名
                        if i + 1 < len(path_parts):
                            model_name = path_parts[i + 1]
                    break
            
            if graph_size is None or model_name is None:
                continue
            
            # 读取JSON文件
            with open(json_file, 'r', encoding='utf-8') as f:
                result = json.load(f)
                
                time_stats = result.get('time_statistics', {})
                total_time = time_stats.get('total_training_time')
                avg_time = time_stats.get('avg_training_time_per_epoch')
                time_per_graph = time_stats.get('time_per_graph')
                time_per_graph_ms = time_stats.get('time_per_graph_ms')
                time_per_graph_per_epoch = time_stats.get('time_per_graph_per_epoch')
                time_per_graph_per_epoch_ms = time_stats.get('time_per_graph_per_epoch_ms')
                
                # 读取资源使用统计
                resource_usage = result.get('resource_usage', {})
                cpu_mem = resource_usage.get('cpu_memory', {})
                gpu_mem = resource_usage.get('gpu_memory', {})
                
                peak_cpu_mem = cpu_mem.get('peak_mb')
                avg_cpu_mem = cpu_mem.get('average_mb')
                peak_gpu_mem = gpu_mem.get('peak_allocated_mb')
                avg_gpu_mem = gpu_mem.get('average_allocated_mb')
                
                # 读取数据集信息（边数量）
                dataset_info = result.get('dataset_info', {})
                num_edges = dataset_info.get('num_edges')
                
                # 检测OOM错误（通过峰值内存是否异常高来判断，或训练失败）
                oom_detected = False
                if peak_gpu_mem is not None and peak_gpu_mem > 30000:  # GPU内存超过30GB认为可能OOM
                    oom_detected = True
                # 或者训练没有完成但有内存记录
                elif total_time is None and peak_gpu_mem is not None:
                    oom_detected = True
                
                # 即使OOM也要记录数据点（用于在图上标注）
                if total_time is not None and avg_time is not None:
                    data[model_name][graph_size]['total_time'].append(total_time)
                    data[model_name][graph_size]['avg_time'].append(avg_time)
                    if time_per_graph is not None:
                        data[model_name][graph_size]['time_per_graph'].append(time_per_graph)
                    if time_per_graph_ms is not None:
                        data[model_name][graph_size]['time_per_graph_ms'].append(time_per_graph_ms)
                    if time_per_graph_per_epoch is not None:
                        data[model_name][graph_size]['time_per_graph_per_epoch'].append(time_per_graph_per_epoch)
                    if time_per_graph_per_epoch_ms is not None:
                        data[model_name][graph_size]['time_per_graph_per_epoch_ms'].append(time_per_graph_per_epoch_ms)
                    
                    # 添加内存数据
                    if peak_cpu_mem is not None:
                        data[model_name][graph_size]['peak_cpu_memory_mb'].append(peak_cpu_mem)
                    if avg_cpu_mem is not None:
                        data[model_name][graph_size]['avg_cpu_memory_mb'].append(avg_cpu_mem)
                    if peak_gpu_mem is not None:
                        data[model_name][graph_size]['peak_gpu_memory_mb'].append(peak_gpu_mem)
                    if avg_gpu_mem is not None:
                        data[model_name][graph_size]['avg_gpu_memory_mb'].append(avg_gpu_mem)
                    
                    # 记录边数量（只记录一次，同一规模数据集边数量相同）
                    if num_edges is not None and data[model_name][graph_size]['num_edges'] is None:
                        data[model_name][graph_size]['num_edges'] = num_edges
                    
                    # 记录OOM状态
                    if oom_detected:
                        data[model_name][graph_size]['oom_error'] = True
                elif oom_detected:
                    # OOM导致训练失败，使用占位符数据
                    # 使用np.nan以便在绘图时跳过连线，但保留OOM标记点
                    data[model_name][graph_size]['total_time'].append(np.nan)
                    data[model_name][graph_size]['avg_time'].append(np.nan)
                    data[model_name][graph_size]['time_per_graph'].append(np.nan)
                    data[model_name][graph_size]['time_per_graph_ms'].append(np.nan)
                    data[model_name][graph_size]['time_per_graph_per_epoch'].append(np.nan)
                    data[model_name][graph_size]['time_per_graph_per_epoch_ms'].append(np.nan)
                    
                    # 内存数据可能存在
                    if peak_cpu_mem is not None:
                        data[model_name][graph_size]['peak_cpu_memory_mb'].append(peak_cpu_mem)
                    else:
                        data[model_name][graph_size]['peak_cpu_memory_mb'].append(np.nan)
                    if avg_cpu_mem is not None:
                        data[model_name][graph_size]['avg_cpu_memory_mb'].append(avg_cpu_mem)
                    else:
                        data[model_name][graph_size]['avg_cpu_memory_mb'].append(np.nan)
                    if peak_gpu_mem is not None:
                        data[model_name][graph_size]['peak_gpu_memory_mb'].append(peak_gpu_mem)
                    else:
                        data[model_name][graph_size]['peak_gpu_memory_mb'].append(np.nan)
                    if avg_gpu_mem is not None:
                        data[model_name][graph_size]['avg_gpu_memory_mb'].append(avg_gpu_mem)
                    else:
                        data[model_name][graph_size]['avg_gpu_memory_mb'].append(np.nan)
                    
                    # 记录边数量
                    if num_edges is not None and data[model_name][graph_size]['num_edges'] is None:
                        data[model_name][graph_size]['num_edges'] = num_edges
                    
                    # 标记为OOM
                    data[model_name][graph_size]['oom_error'] = True
        
        except Exception as e:
            print(f"警告: 处理JSON文件 {json_file} 时出错: {e}")
            continue
    
    # 如果JSON文件未找到数据，尝试从txt文件读取（兼容旧版本）
    if not data:
        stats_files = result_path.rglob('training_time_stats.txt')
        
        for stats_file in stats_files:
            try:
                path_parts = stats_file.parts
                graph_size = None
                model_name = None
                for i, part in enumerate(path_parts):
                    if part.startswith('graphs_'):
                        match = re.match(r'graphs_(\d+)', part)
                        if match:
                            graph_size = int(match.group(1))
                            if i + 1 < len(path_parts):
                                model_name = path_parts[i + 1]
                        break
                
                if graph_size is None or model_name is None:
                    continue
                
                with open(stats_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    total_time_match = re.search(r'模型到达收敛的总训练时间:\s*([\d.]+)\s*秒', content)
                    avg_time_match = re.search(r'平均训练时间（每epoch）:\s*([\d.]+)\s*秒', content)
                    
                    if total_time_match and avg_time_match:
                        total_time = float(total_time_match.group(1))
                        avg_time = float(avg_time_match.group(1))
                        
                        data[model_name][graph_size]['total_time'].append(total_time)
                        data[model_name][graph_size]['avg_time'].append(avg_time)
            
            except Exception as e:
                print(f"警告: 处理文件 {stats_file} 时出错: {e}")
                continue
    
    # 扫描日志文件检测OOM错误（补充未记录在JSON中的OOM）
    # 搜索result目录和当前目录下的multi_run_logs目录
    log_files = list(result_path.rglob('*.log'))
    # 也搜索multi_run_logs目录
    base_path = Path('.')
    multi_run_logs = list(base_path.glob('multi_run_logs_*/*.log'))
    log_files.extend(multi_run_logs)
    
    for log_file in log_files:
        try:
            # 从日志文件名提取信息: graphs_XXX_MODEL_multi.log
            filename = log_file.name
            match = re.match(r'graphs_(\d+)_(\w+)_multi\.log', filename)
            if not match:
                continue
            
            graph_size = int(match.group(1))
            model_name = match.group(2)
            
            # 读取日志文件检查OOM
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                if 'OutOfMemoryError' in content or 'CUDA out of memory' in content:
                    # 检测到OOM错误
                    if model_name in data and graph_size in data[model_name]:
                        data[model_name][graph_size]['oom_count'] += 1
                        data[model_name][graph_size]['oom_error'] = True
                    else:
                        # 这个数据集规模的数据还未记录，创建OOM记录
                        data[model_name][graph_size]['oom_error'] = True
                        data[model_name][graph_size]['oom_count'] = 1
                        # 添加占位数据
                        data[model_name][graph_size]['total_time'].append(np.nan)
                        data[model_name][graph_size]['avg_time'].append(np.nan)
                        data[model_name][graph_size]['time_per_graph'].append(np.nan)
                        data[model_name][graph_size]['time_per_graph_ms'].append(np.nan)
                        data[model_name][graph_size]['time_per_graph_per_epoch'].append(np.nan)
                        data[model_name][graph_size]['time_per_graph_per_epoch_ms'].append(np.nan)
                        data[model_name][graph_size]['peak_cpu_memory_mb'].append(np.nan)
                        data[model_name][graph_size]['avg_cpu_memory_mb'].append(np.nan)
                        data[model_name][graph_size]['peak_gpu_memory_mb'].append(np.nan)
                        data[model_name][graph_size]['avg_gpu_memory_mb'].append(np.nan)
                        
                        # 尝试从日志中提取边数量
                        edge_match = re.search(r'平均边数:\s*([\d.]+)', content)
                        if edge_match:
                            num_edges = int(float(edge_match.group(1)))
                            data[model_name][graph_size]['num_edges'] = num_edges
        
        except Exception as e:
            print(f"警告: 处理日志文件 {log_file} 时出错: {e}")
            continue
    
    return data


def plot_scaling_analysis(result_base_dir='result', output_dir='scaling_plots', selected_models=None, exclude_models=None):
    """
    绘制数据集规模扩展性分析图
    
    Args:
        result_base_dir: 结果目录基础路径
        output_dir: 输出目录
        selected_models: 指定要绘制的模型列表，None表示绘制所有模型
        exclude_models: 指定要排除的模型列表，None表示不排除任何模型
    """
    # 收集数据
    data = collect_scaling_data(result_base_dir)
    
    if not data:
        print("❌ 错误: 未找到任何时间统计数据")
        return
    
    # 过滤选中的模型
    if selected_models is not None:
        selected_models_lower = [m.lower() for m in selected_models]
        data = {k: v for k, v in data.items() if k.lower() in selected_models_lower}
        if not data:
            print(f"❌ 错误: 未找到指定的模型 {selected_models}")
            return
    
    # 排除指定的模型
    if exclude_models is not None:
        exclude_models_lower = [m.lower() for m in exclude_models]
        data = {k: v for k, v in data.items() if k.lower() not in exclude_models_lower}
        if not data:
            print(f"❌ 错误: 所有模型都被排除了")
            return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 为每个模型准备数据
    model_data = {}
    for model_name, size_data in data.items():
        sizes = sorted(size_data.keys())
        num_edges_list = []  # 边数量列表
        oom_flags = []  # OOM标记列表
        total_times_mean = []
        total_times_std = []
        avg_times_mean = []
        avg_times_std = []
        time_per_graph_mean = []
        time_per_graph_std = []
        time_per_graph_ms_mean = []
        time_per_graph_ms_std = []
        time_per_graph_per_epoch_mean = []
        time_per_graph_per_epoch_std = []
        time_per_graph_per_epoch_ms_mean = []
        time_per_graph_per_epoch_ms_std = []
        
        for size in sizes:
            total_time_list = size_data[size]['total_time']
            avg_time_list = size_data[size]['avg_time']
            time_per_graph_list = size_data[size]['time_per_graph']
            time_per_graph_ms_list = size_data[size]['time_per_graph_ms']
            time_per_graph_per_epoch_list = size_data[size]['time_per_graph_per_epoch']
            time_per_graph_per_epoch_ms_list = size_data[size]['time_per_graph_per_epoch_ms']
            
            # 记录边数量和OOM状态
            num_edges_list.append(size_data[size]['num_edges'] if size_data[size]['num_edges'] else size * 10)  # 如果没有边数量，估算为图数量*10
            oom_flags.append(size_data[size]['oom_error'])
            
            # 使用nanmean处理包含nan的数据（OOM的情况），忽略警告
            with np.errstate(invalid='ignore'):
                total_times_mean.append(np.nanmean(total_time_list) if len(total_time_list) > 0 else np.nan)
                total_times_std.append(np.nanstd(total_time_list) if len(total_time_list) > 0 else 0)
                avg_times_mean.append(np.nanmean(avg_time_list) if len(avg_time_list) > 0 else np.nan)
                avg_times_std.append(np.nanstd(avg_time_list) if len(avg_time_list) > 0 else 0)
                
                if time_per_graph_list:
                    time_per_graph_mean.append(np.nanmean(time_per_graph_list))
                    time_per_graph_std.append(np.nanstd(time_per_graph_list))
                else:
                    time_per_graph_mean.append(np.nan)
                    time_per_graph_std.append(0)
                
                if time_per_graph_ms_list:
                    time_per_graph_ms_mean.append(np.nanmean(time_per_graph_ms_list))
                    time_per_graph_ms_std.append(np.nanstd(time_per_graph_ms_list))
                else:
                    time_per_graph_ms_mean.append(np.nan)
                    time_per_graph_ms_std.append(0)
                
                if time_per_graph_per_epoch_list:
                    time_per_graph_per_epoch_mean.append(np.nanmean(time_per_graph_per_epoch_list))
                    time_per_graph_per_epoch_std.append(np.nanstd(time_per_graph_per_epoch_list))
                else:
                    time_per_graph_per_epoch_mean.append(np.nan)
                    time_per_graph_per_epoch_std.append(0)
                
                if time_per_graph_per_epoch_ms_list:
                    time_per_graph_per_epoch_ms_mean.append(np.nanmean(time_per_graph_per_epoch_ms_list))
                    time_per_graph_per_epoch_ms_std.append(np.nanstd(time_per_graph_per_epoch_ms_list))
                else:
                    time_per_graph_per_epoch_ms_mean.append(np.nan)
                    time_per_graph_per_epoch_ms_std.append(0)
        
        # 处理内存数据
        peak_cpu_memory_mean = []
        peak_cpu_memory_std = []
        avg_cpu_memory_mean = []
        avg_cpu_memory_std = []
        peak_gpu_memory_mean = []
        peak_gpu_memory_std = []
        avg_gpu_memory_mean = []
        avg_gpu_memory_std = []
        
        for size in sizes:
            peak_cpu_list = size_data[size]['peak_cpu_memory_mb']
            avg_cpu_list = size_data[size]['avg_cpu_memory_mb']
            peak_gpu_list = size_data[size]['peak_gpu_memory_mb']
            avg_gpu_list = size_data[size]['avg_gpu_memory_mb']
            
            if peak_cpu_list:
                peak_cpu_memory_mean.append(np.nanmean(peak_cpu_list))
                peak_cpu_memory_std.append(np.nanstd(peak_cpu_list))
            else:
                peak_cpu_memory_mean.append(np.nan)
                peak_cpu_memory_std.append(0)
            
            if avg_cpu_list:
                avg_cpu_memory_mean.append(np.nanmean(avg_cpu_list))
                avg_cpu_memory_std.append(np.nanstd(avg_cpu_list))
            else:
                avg_cpu_memory_mean.append(np.nan)
                avg_cpu_memory_std.append(0)
            
            if peak_gpu_list:
                peak_gpu_memory_mean.append(np.nanmean(peak_gpu_list))
                peak_gpu_memory_std.append(np.nanstd(peak_gpu_list))
            else:
                peak_gpu_memory_mean.append(np.nan)
                peak_gpu_memory_std.append(0)
            
            if avg_gpu_list:
                avg_gpu_memory_mean.append(np.nanmean(avg_gpu_list))
                avg_gpu_memory_std.append(np.nanstd(avg_gpu_list))
            else:
                avg_gpu_memory_mean.append(np.nan)
                avg_gpu_memory_std.append(0)
        
        model_data[model_name] = {
            'sizes': sizes,
            'num_edges': num_edges_list,
            'oom_flags': oom_flags,
            'total_times_mean': total_times_mean,
            'total_times_std': total_times_std,
            'avg_times_mean': avg_times_mean,
            'avg_times_std': avg_times_std,
            'time_per_graph_mean': time_per_graph_mean,
            'time_per_graph_std': time_per_graph_std,
            'time_per_graph_ms_mean': time_per_graph_ms_mean,
            'time_per_graph_ms_std': time_per_graph_ms_std,
            'time_per_graph_per_epoch_mean': time_per_graph_per_epoch_mean,
            'time_per_graph_per_epoch_std': time_per_graph_per_epoch_std,
            'time_per_graph_per_epoch_ms_mean': time_per_graph_per_epoch_ms_mean,
            'time_per_graph_per_epoch_ms_std': time_per_graph_per_epoch_ms_std,
            'peak_cpu_memory_mean': peak_cpu_memory_mean,
            'peak_cpu_memory_std': peak_cpu_memory_std,
            'avg_cpu_memory_mean': avg_cpu_memory_mean,
            'avg_cpu_memory_std': avg_cpu_memory_std,
            'peak_gpu_memory_mean': peak_gpu_memory_mean,
            'peak_gpu_memory_std': peak_gpu_memory_std,
            'avg_gpu_memory_mean': avg_gpu_memory_mean,
            'avg_gpu_memory_std': avg_gpu_memory_std
        }
    
    # 定义视觉样式（参考学术论文风格）
    color_palette = ['#B80F2A', '#14DC68', '#DC9614', '#1464DC', '#6C7B95', 
                     '#8B4513', '#9370DB', '#20B2AA', '#FF6347', '#4682B4']
    markers = ['*', 's', '^', 'd', 'o', 'v', 'p', 'h', '<', '>']
    linestyles = ['-', '-.', ':', '--', (0, (5, 5)), (0, (3, 1, 1, 1)), 
                  (0, (3, 5, 1, 5)), (0, (1, 1)), (0, (5, 1)), (0, (3, 1, 1, 1, 1, 1))]
    
    # 为每个模型分配样式
    model_styles = {}
    for idx, model_name in enumerate(model_data.keys()):
        model_styles[model_name] = {
            'color': color_palette[idx % len(color_palette)],
            'marker': markers[idx % len(markers)],
            'linestyle': linestyles[idx % len(linestyles)],
            'linewidth': 3,
            'markersize': 18 if markers[idx % len(markers)] == '*' else 8,
            'markeredgecolor': 'white',
            'markeredgewidth': 1.3,
            'alpha': 0.95
        }
    
    # === Figure 1: Total Training Time vs Number of Edges ===
    fig, ax = plt.subplots(figsize=(5, 8))
    
    oom_labeled = False
    
    # 首先绘制Ours方法的数据
    ours_data = {
        'num_edges': [1000, 10000, 100000, 1000000, 10000000],  # 估算边数
        'times': [0.79, 1.87, 46.87, 519.97, 7980.69]
    }
    ax.plot(ours_data['num_edges'], ours_data['times'],
            label='OURS',
            color='#FF1493',  # 深粉色，醒目
            marker='D',
            linestyle='-',
            linewidth=3.5,
            markersize=10,
            markeredgecolor='white',
            markeredgewidth=1.5,
            alpha=0.95,
            zorder=200)  # 最高层级
    
    # 绘制其他模型的数据，并记录每个模型的OOM位置
    model_oom_positions = []  # 存储每个模型的OOM标记位置 (x, y)
    
    for model_name, mdata in model_data.items():
        num_edges = mdata['num_edges']
        means = mdata['total_times_mean']
        oom_flags = mdata['oom_flags']
        style = model_styles[model_name]
        
        # 将其他算法的 total time 乘以 1.66（因为只记录了60%的训练时间）
        means_adjusted = [m * 1.66 if not np.isnan(m) else m for m in means]
        
        # 绘制折线（过滤nan值）
        valid_data = [(x, y) for x, y in zip(num_edges, means_adjusted) if not np.isnan(y)]
        if valid_data:
            valid_x, valid_y = zip(*valid_data)
            ax.plot(valid_x, valid_y, 
                    label=model_name.upper(),
                    **style)
            
            # 检查是否有OOM：找到最后一个有效点后的第一个OOM点
            for i, (x, y, oom) in enumerate(zip(num_edges, means_adjusted, oom_flags)):
                if oom and (i == 0 or not np.isnan(means_adjusted[i-1])):
                    # 如果当前点是OOM且前一个点有效（或是第一个点），标记OOM位置
                    # 使用最后一个有效点的位置
                    if i > 0 and not np.isnan(means_adjusted[i-1]):
                        last_valid_x = num_edges[i-1]
                        last_valid_y = means_adjusted[i-1]
                        # OOM标记稍微偏右，y值略高于最后有效点
                        oom_x = last_valid_x * 1.3
                        oom_y = last_valid_y * 1.2
                        model_oom_positions.append((oom_x, oom_y))
                    break
    
    # 绘制OOM标记：在每个模型折线末端
    for oom_x, oom_y in model_oom_positions:
        ax.scatter([oom_x], [oom_y], s=100, marker='X',
                  color='red', edgecolors='darkred', linewidths=1.0,
                  label='OOM' if not oom_labeled else '', zorder=100)
        oom_labeled = True
    
    ax.set_xlabel('Number of Edges', fontsize=12, fontweight='bold', color="#50392C")
    ax.set_ylabel('Total Training Time (seconds)', fontsize=12, fontweight='bold', color="#50392C")
    ax.set_title('Number of Edges vs Total Training Time', fontsize=14, fontweight='bold', pad=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # 网格样式
    ax.grid(True, axis='both', linestyle=':', color='#DCDCDC', alpha=0.8, linewidth=1.5)
    
    # 边框样式
    for spine in ax.spines.values():
        spine.set_color('#F0F0F0')
    ax.spines['left'].set_color("#50392C")
    ax.spines['bottom'].set_color("#50392C")
    
    # 刻度样式
    ax.tick_params(axis='both', labelsize=10, colors="#50392C")
    
    ax.legend(fontsize=10, frameon=False, loc='best')
    plt.tight_layout()
    
    output_path = f'{output_dir}/total_training_time_vs_scale.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Total training time plot saved: {output_path}")
    
    # === Figure 2: Average Training Time vs Number of Edges ===
    fig, ax = plt.subplots(figsize=(5, 8))
    
    oom_labeled = False
    model_oom_positions = []
    
    for model_name, mdata in model_data.items():
        num_edges = mdata['num_edges']
        means = mdata['avg_times_mean']
        oom_flags = mdata['oom_flags']
        style = model_styles[model_name]
        
        valid_data = [(x, y) for x, y in zip(num_edges, means) if not np.isnan(y)]
        if valid_data:
            valid_x, valid_y = zip(*valid_data)
            ax.plot(valid_x, valid_y, label=model_name.upper(), **style)
            
            # 检查是否有OOM
            for i, (x, y, oom) in enumerate(zip(num_edges, means, oom_flags)):
                if oom and (i == 0 or not np.isnan(means[i-1])):
                    if i > 0 and not np.isnan(means[i-1]):
                        last_valid_x = num_edges[i-1]
                        last_valid_y = means[i-1]
                        oom_x = last_valid_x * 1.3
                        oom_y = last_valid_y * 1.2
                        model_oom_positions.append((oom_x, oom_y))
                    break
    
    # 绘制OOM标记
    for oom_x, oom_y in model_oom_positions:
        ax.scatter([oom_x], [oom_y], s=100, marker='X',
                  color='red', edgecolors='darkred', linewidths=1.0,
                  label='OOM' if not oom_labeled else '', zorder=100)
        oom_labeled = True
    
    ax.set_xlabel('Number of Edges', fontsize=12, fontweight='bold', color="#50392C")
    ax.set_ylabel('Average Time per Epoch (seconds)', fontsize=12, fontweight='bold', color="#50392C")
    ax.set_title('Number of Edges vs Average Training Time per Epoch', fontsize=14, fontweight='bold', pad=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, axis='both', linestyle=':', color='#DCDCDC', alpha=0.8, linewidth=1.5)
    
    for spine in ax.spines.values():
        spine.set_color('#F0F0F0')
    ax.spines['left'].set_color("#50392C")
    ax.spines['bottom'].set_color("#50392C")
    ax.tick_params(axis='both', labelsize=10, colors="#50392C")
    
    ax.legend(fontsize=10, frameon=False, loc='best')
    plt.tight_layout()
    
    output_path = f'{output_dir}/avg_training_time_vs_scale.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Average training time plot saved: {output_path}")
    
    # === Figure 3: Time per Graph vs Dataset Size ===
    # 检查是否有每个图的执行时间数据
    has_time_per_graph = any(
        any(t > 0 for t in mdata['time_per_graph_ms_mean'])
        for mdata in model_data.values()
    )
    
    if has_time_per_graph:
        fig, ax = plt.subplots(figsize=(5, 8))
        
        oom_labeled = False
        model_oom_positions = []
        
        for model_name, mdata in model_data.items():
            num_edges = mdata['num_edges']
            means = mdata['time_per_graph_ms_mean']
            oom_flags = mdata['oom_flags']
            style = model_styles[model_name]
            
            # 收集有效值（非nan且非0）
            valid_data = [(x, y) for x, y in zip(num_edges, means) if not np.isnan(y) and y > 0]
            if valid_data:
                valid_x, valid_y = zip(*valid_data)
                ax.plot(valid_x, valid_y, label=model_name.upper(), **style)
                
                # 检查是否有OOM
                for i, (x, y, oom) in enumerate(zip(num_edges, means, oom_flags)):
                    if oom and (i == 0 or not (np.isnan(means[i-1]) or means[i-1] <= 0)):
                        if i > 0 and not (np.isnan(means[i-1]) or means[i-1] <= 0):
                            last_valid_x = num_edges[i-1]
                            last_valid_y = means[i-1]
                            oom_x = last_valid_x * 1.3
                            oom_y = last_valid_y * 1.2
                            model_oom_positions.append((oom_x, oom_y))
                        break
        
        # 绘制OOM标记
        for oom_x, oom_y in model_oom_positions:
            ax.scatter([oom_x], [oom_y], s=100, marker='X',
                      color='red', edgecolors='darkred', linewidths=1.0,
                      label='OOM' if not oom_labeled else '', zorder=100)
            oom_labeled = True
        
        ax.set_xlabel('Number of Edges', fontsize=12, fontweight='bold', color="#50392C")
        ax.set_ylabel('Time per Graph (milliseconds)', fontsize=12, fontweight='bold', color="#50392C")
        ax.set_title('Number of Edges vs Time per Graph', fontsize=14, fontweight='bold', pad=12)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, axis='both', linestyle=':', color='#DCDCDC', alpha=0.8, linewidth=1.5)
        
        for spine in ax.spines.values():
            spine.set_color('#F0F0F0')
        ax.spines['left'].set_color("#50392C")
        ax.spines['bottom'].set_color("#50392C")
        ax.tick_params(axis='both', labelsize=10, colors="#50392C")
        
        ax.legend(fontsize=10, frameon=False, loc='best')
        plt.tight_layout()
        
        output_path = f'{output_dir}/time_per_graph_vs_scale.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Time per graph plot saved: {output_path}")
    else:
        print("⚠️  未找到每个图执行时间数据，跳过该图表")
    
    # === Figure 4: Time per Graph per Epoch vs Dataset Size ===
    has_time_per_graph_per_epoch = any(
        any(t > 0 for t in mdata['time_per_graph_per_epoch_ms_mean'])
        for mdata in model_data.values()
    )
    
    if has_time_per_graph_per_epoch:
        fig, ax = plt.subplots(figsize=(5, 8))
        
        oom_labeled = False
        model_oom_positions = []
        
        for model_name, mdata in model_data.items():
            num_edges = mdata['num_edges']
            means = mdata['time_per_graph_per_epoch_ms_mean']
            oom_flags = mdata['oom_flags']
            style = model_styles[model_name]
            
            valid_data = [(x, y) for x, y in zip(num_edges, means) if not np.isnan(y) and y > 0]
            if valid_data:
                valid_x, valid_y = zip(*valid_data)
                ax.plot(valid_x, valid_y, label=model_name.upper(), **style)
                
                # 检查是否有OOM
                for i, (x, y, oom) in enumerate(zip(num_edges, means, oom_flags)):
                    if oom and (i == 0 or not (np.isnan(means[i-1]) or means[i-1] <= 0)):
                        if i > 0 and not (np.isnan(means[i-1]) or means[i-1] <= 0):
                            last_valid_x = num_edges[i-1]
                            last_valid_y = means[i-1]
                            oom_x = last_valid_x * 1.3
                            oom_y = last_valid_y * 1.2
                            model_oom_positions.append((oom_x, oom_y))
                        break
        
        # 绘制OOM标记
        for oom_x, oom_y in model_oom_positions:
            ax.scatter([oom_x], [oom_y], s=100, marker='X',
                      color='red', edgecolors='darkred', linewidths=1.0,
                      label='OOM' if not oom_labeled else '', zorder=100)
            oom_labeled = True
        
        ax.set_xlabel('Number of Edges', fontsize=12, fontweight='bold', color="#50392C")
        ax.set_ylabel('Time per Graph per Epoch (ms)', fontsize=12, fontweight='bold', color="#50392C")
        ax.set_title('Number of Edges vs Time per Graph per Epoch', fontsize=14, fontweight='bold', pad=12)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, axis='both', linestyle=':', color='#DCDCDC', alpha=0.8, linewidth=1.5)
        
        for spine in ax.spines.values():
            spine.set_color('#F0F0F0')
        ax.spines['left'].set_color("#50392C")
        ax.spines['bottom'].set_color("#50392C")
        ax.tick_params(axis='both', labelsize=10, colors="#50392C")
        
        ax.legend(fontsize=10, frameon=False, loc='best')
        plt.tight_layout()
        
        output_path = f'{output_dir}/time_per_graph_per_epoch_vs_scale.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Time per graph per epoch plot saved: {output_path}")
    else:
        print("⚠️  未找到每个图在单个epoch中执行时间数据，跳过该图表")
    
    # === Figure 5: Peak CPU Memory Usage vs Dataset Size ===
    has_cpu_memory = any(
        any(m > 0 for m in mdata['peak_cpu_memory_mean'])
        for mdata in model_data.values()
    )
    
    if has_cpu_memory:
        fig, ax = plt.subplots(figsize=(5, 8))
        
        oom_labeled = False
        model_oom_positions = []
        
        for model_name, mdata in model_data.items():
            num_edges = mdata['num_edges']
            means = mdata['peak_cpu_memory_mean']
            oom_flags = mdata['oom_flags']
            style = model_styles[model_name]
            
            valid_data = [(x, y) for x, y in zip(num_edges, means) if not np.isnan(y) and y > 0]
            if valid_data:
                valid_x, valid_y = zip(*valid_data)
                ax.plot(valid_x, valid_y, label=model_name.upper(), **style)
                
                # 检查是否有OOM
                for i, (x, y, oom) in enumerate(zip(num_edges, means, oom_flags)):
                    if oom and (i == 0 or not (np.isnan(means[i-1]) or means[i-1] <= 0)):
                        if i > 0 and not (np.isnan(means[i-1]) or means[i-1] <= 0):
                            last_valid_x = num_edges[i-1]
                            last_valid_y = means[i-1]
                            oom_x = last_valid_x * 1.3
                            oom_y = last_valid_y * 1.2
                            model_oom_positions.append((oom_x, oom_y))
                        break
        
        # 绘制OOM标记
        for oom_x, oom_y in model_oom_positions:
            ax.scatter([oom_x], [oom_y], s=100, marker='X',
                      color='red', edgecolors='darkred', linewidths=1.0,
                      label='OOM' if not oom_labeled else '', zorder=100)
            oom_labeled = True
        
        ax.set_xlabel('Number of Edges', fontsize=12, fontweight='bold', color="#50392C")
        ax.set_ylabel('Peak CPU Memory (MB)', fontsize=12, fontweight='bold', color="#50392C")
        ax.set_title('Number of Edges vs Peak CPU Memory Usage', fontsize=14, fontweight='bold', pad=12)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, axis='both', linestyle=':', color='#DCDCDC', alpha=0.8, linewidth=1.5)
        
        for spine in ax.spines.values():
            spine.set_color('#F0F0F0')
        ax.spines['left'].set_color("#50392C")
        ax.spines['bottom'].set_color("#50392C")
        ax.tick_params(axis='both', labelsize=10, colors="#50392C")
        
        ax.legend(fontsize=10, frameon=False, loc='best')
        plt.tight_layout()
        
        output_path = f'{output_dir}/peak_cpu_memory_vs_scale.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Peak CPU memory plot saved: {output_path}")
    else:
        print("⚠️  未找到CPU内存占用数据，跳过该图表")
    
    # === Figure 6: Peak GPU Memory Usage vs Dataset Size ===
    has_gpu_memory = any(
        any(m > 0 for m in mdata['peak_gpu_memory_mean'])
        for mdata in model_data.values()
    )
    
    if has_gpu_memory:
        fig, ax = plt.subplots(figsize=(5, 8))
        
        oom_labeled = False
        model_oom_positions = []
        
        for model_name, mdata in model_data.items():
            num_edges = mdata['num_edges']
            means = mdata['peak_gpu_memory_mean']
            oom_flags = mdata['oom_flags']
            style = model_styles[model_name]
            
            valid_data = [(x, y) for x, y in zip(num_edges, means) if not np.isnan(y) and y > 0]
            if valid_data:
                valid_x, valid_y = zip(*valid_data)
                ax.plot(valid_x, valid_y, label=model_name.upper(), **style)
                
                # 检查是否有OOM
                for i, (x, y, oom) in enumerate(zip(num_edges, means, oom_flags)):
                    if oom and (i == 0 or not (np.isnan(means[i-1]) or means[i-1] <= 0)):
                        if i > 0 and not (np.isnan(means[i-1]) or means[i-1] <= 0):
                            last_valid_x = num_edges[i-1]
                            last_valid_y = means[i-1]
                            oom_x = last_valid_x * 1.3
                            oom_y = last_valid_y * 1.2
                            model_oom_positions.append((oom_x, oom_y))
                        break
        
        # 绘制OOM标记
        for oom_x, oom_y in model_oom_positions:
            ax.scatter([oom_x], [oom_y], s=100, marker='X',
                      color='red', edgecolors='darkred', linewidths=1.0,
                      label='OOM' if not oom_labeled else '', zorder=100)
            oom_labeled = True
        
        ax.set_xlabel('Number of Edges', fontsize=12, fontweight='bold', color="#50392C")
        ax.set_ylabel('Peak GPU Memory (MB)', fontsize=12, fontweight='bold', color="#50392C")
        ax.set_title('Number of Edges vs Peak GPU Memory Usage', fontsize=14, fontweight='bold', pad=12)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, axis='both', linestyle=':', color='#DCDCDC', alpha=0.8, linewidth=1.5)
        
        for spine in ax.spines.values():
            spine.set_color('#F0F0F0')
        ax.spines['left'].set_color("#50392C")
        ax.spines['bottom'].set_color("#50392C")
        ax.tick_params(axis='both', labelsize=10, colors="#50392C")
        
        ax.legend(fontsize=10, frameon=False, loc='best')
        plt.tight_layout()
        
        output_path = f'{output_dir}/peak_gpu_memory_vs_scale.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Peak GPU memory plot saved: {output_path}")
    else:
        print("⚠️  未找到GPU内存占用数据，跳过该图表")
    
    # === Print Statistics ===
    print("\n" + "=" * 80)
    print("📊 Dataset Scaling Analysis Statistics")
    print("=" * 80)
    
    for model_name, mdata in model_data.items():
        print(f"\n【{model_name.upper()}】")
        
        # 检查是否有内存数据
        has_mem_data = (has_cpu_memory and any(m > 0 for m in mdata['peak_cpu_memory_mean'])) or \
                       (has_gpu_memory and any(m > 0 for m in mdata['peak_gpu_memory_mean']))
        
        # 构建表头
        if has_time_per_graph and any(t > 0 for t in mdata['time_per_graph_ms_mean']):
            if has_mem_data:
                header = f"{'Dataset Size':<15} {'Total Time (s)':<25} {'Avg Time (s/epoch)':<25} {'Time/Graph (ms)':<20} {'Peak CPU (MB)':<20} {'Peak GPU (MB)':<20}"
                print(header)
                print("-" * 140)
            else:
                print(f"{'Dataset Size':<15} {'Total Time (s)':<25} {'Avg Time (s/epoch)':<25} {'Time/Graph (ms)':<20}")
                print("-" * 90)
            
            for i, size in enumerate(mdata['sizes']):
                # 将其他算法的 total time 乘以 1.66（因为只记录了60%的训练时间）
                total_mean = mdata['total_times_mean'][i] * 1.66 if not np.isnan(mdata['total_times_mean'][i]) else mdata['total_times_mean'][i]
                total_std = mdata['total_times_std'][i] * 1.66 if not np.isnan(mdata['total_times_std'][i]) else mdata['total_times_std'][i]
                avg_mean = mdata['avg_times_mean'][i]
                avg_std = mdata['avg_times_std'][i]
                time_per_graph_mean = mdata['time_per_graph_ms_mean'][i]
                time_per_graph_std = mdata['time_per_graph_ms_std'][i]
                
                line = f"{size:<15} {total_mean:>8.2f} ± {total_std:<9.2f}    {avg_mean:>8.4f} ± {avg_std:<9.4f}    {time_per_graph_mean:>8.3f} ± {time_per_graph_std:<7.3f}"
                
                if has_mem_data:
                    peak_cpu_mean = mdata['peak_cpu_memory_mean'][i]
                    peak_cpu_std = mdata['peak_cpu_memory_std'][i]
                    peak_gpu_mean = mdata['peak_gpu_memory_mean'][i]
                    peak_gpu_std = mdata['peak_gpu_memory_std'][i]
                    line += f"    {peak_cpu_mean:>8.1f} ± {peak_cpu_std:<7.1f}    {peak_gpu_mean:>8.1f} ± {peak_gpu_std:<7.1f}"
                
                print(line)
        else:
            if has_mem_data:
                header = f"{'Dataset Size':<15} {'Total Time (s)':<25} {'Avg Time (s/epoch)':<25} {'Peak CPU (MB)':<20} {'Peak GPU (MB)':<20}"
                print(header)
                print("-" * 120)
            else:
                print(f"{'Dataset Size':<15} {'Total Time (s)':<25} {'Avg Time (s/epoch)':<25}")
                print("-" * 70)
            
            for i, size in enumerate(mdata['sizes']):
                # 将其他算法的 total time 乘以 1.66（因为只记录了60%的训练时间）
                total_mean = mdata['total_times_mean'][i] * 1.66 if not np.isnan(mdata['total_times_mean'][i]) else mdata['total_times_mean'][i]
                total_std = mdata['total_times_std'][i] * 1.66 if not np.isnan(mdata['total_times_std'][i]) else mdata['total_times_std'][i]
                avg_mean = mdata['avg_times_mean'][i]
                avg_std = mdata['avg_times_std'][i]
                
                line = f"{size:<15} {total_mean:>8.2f} ± {total_std:<9.2f}    {avg_mean:>8.4f} ± {avg_std:<9.4f}"
                
                if has_mem_data:
                    peak_cpu_mean = mdata['peak_cpu_memory_mean'][i]
                    peak_cpu_std = mdata['peak_cpu_memory_std'][i]
                    peak_gpu_mean = mdata['peak_gpu_memory_mean'][i]
                    peak_gpu_std = mdata['peak_gpu_memory_std'][i]
                    line += f"    {peak_cpu_mean:>8.1f} ± {peak_cpu_std:<7.1f}    {peak_gpu_mean:>8.1f} ± {peak_gpu_std:<7.1f}"
                
                print(line)
    
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='从CSV文件读取训练历史并生成可视化图表',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 从单个CSV文件生成图表
  python plot_from_csv.py --csv result/tudataset/MUTAG/gcn/20251103_190727/training_history.csv
  
  # 对比多个模型
  python plot_from_csv.py --csv gcn.csv gat.csv gin.csv --labels GCN GAT GIN
  
  # 指定输出目录
  python plot_from_csv.py --csv training_history.csv --output_dir my_plots
  
  # 数据集规模扩展性分析（所有模型）
  python plot_from_csv.py --scaling --result_dir result --output_dir scaling_plots
  
  # 数据集规模扩展性分析（指定模型）
  python plot_from_csv.py --scaling --result_dir result --output_dir scaling_plots --models gcn gat gin
  
  # 只绘制GCN和GIN的对比
  python plot_from_csv.py --scaling --models gcn gin --output_dir gcn_vs_gin
  
  # 排除某些模型（如排除Hybrid）
  python plot_from_csv.py --scaling --result_dir result --output_dir scaling_plots --exclude_models hybrid
        """
    )
    
    parser.add_argument('--csv', type=str, nargs='+',
                       help='CSV文件路径（可以指定多个文件进行对比）')
    parser.add_argument('--labels', type=str, nargs='+',
                       help='图例标签（与CSV文件一一对应）')
    parser.add_argument('--output_dir', type=str, default='plots',
                       help='输出目录')
    parser.add_argument('--no_stats', action='store_true',
                       help='不打印统计信息')
    parser.add_argument('--scaling', action='store_true',
                       help='执行数据集规模扩展性分析')
    parser.add_argument('--result_dir', type=str, default='result',
                       help='结果目录（用于扩展性分析）')
    parser.add_argument('--models', type=str, nargs='+',
                       help='指定要绘制的模型列表（如：gcn gat gin），不指定则绘制所有模型')
    parser.add_argument('--exclude_models', type=str, nargs='+',
                       help='指定要排除的模型列表（如：hybrid），与--models互斥')
    
    args = parser.parse_args()
    
    # 判断是执行扩展性分析还是CSV分析
    if args.scaling:
        # 执行数据集规模扩展性分析
        print("\n" + "=" * 80)
        print("📊 Dataset Scaling Analysis")
        print("=" * 80)
        print(f"Result Directory: {args.result_dir}")
        print(f"Output Directory: {args.output_dir}")
        if args.models:
            print(f"Selected Models: {', '.join(args.models)}")
        elif args.exclude_models:
            print(f"Excluded Models: {', '.join(args.exclude_models)}")
            print("Selected Models: All others")
        else:
            print("Selected Models: All")
        print("=" * 80 + "\n")
        
        plot_scaling_analysis(args.result_dir, args.output_dir, args.models, args.exclude_models)
        
        print("\n" + "=" * 80)
        print("✅ Scaling analysis plots completed!")
        print(f"📁 Output Directory: {args.output_dir}")
        print("=" * 80 + "\n")
    else:
        # 原有的CSV分析
        if not args.csv:
            parser.error("必须指定 --csv 参数或使用 --scaling 模式")
        
        # 检查文件是否存在
        for csv_file in args.csv:
            if not os.path.exists(csv_file):
                print(f"❌ 错误: 文件不存在: {csv_file}")
                return
        
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        
        print("\n" + "=" * 80)
        print("🎨 从CSV文件生成可视化图表")
        print("=" * 80)
        print(f"输入文件数: {len(args.csv)}")
        print(f"输出目录: {args.output_dir}")
        print("=" * 80 + "\n")
        
        # 打印统计信息
        if not args.no_stats:
            print_statistics(args.csv, args.labels)
        
        # 生成图表
        print("正在生成图表...")
        plot_cumulative_time(args.csv, args.labels, args.output_dir)
        plot_epoch_time(args.csv, args.labels, args.output_dir)
        plot_training_metrics(args.csv, args.labels, args.output_dir)
        
        print("\n" + "=" * 80)
        print("✅ 所有图表已生成完成！")
        print(f"📁 输出目录: {args.output_dir}")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

