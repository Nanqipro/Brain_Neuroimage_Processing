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
        """
    )
    
    parser.add_argument('--csv', type=str, nargs='+', required=True,
                       help='CSV文件路径（可以指定多个文件进行对比）')
    parser.add_argument('--labels', type=str, nargs='+',
                       help='图例标签（与CSV文件一一对应）')
    parser.add_argument('--output_dir', type=str, default='plots',
                       help='输出目录')
    parser.add_argument('--no_stats', action='store_true',
                       help='不打印统计信息')
    
    args = parser.parse_args()
    
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

