#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速修复脚本

检查并修复导致CUDA错误的标签问题
"""

import pandas as pd
import numpy as np
import os


def quick_fix_labels(data_path='./data'):
    """
    快速检查和修复标签问题
    
    Parameters
    ----------
    data_path : str
        数据目录路径
    """
    print("=== 快速修复标签问题 ===")
    
    graphs_file = os.path.join(data_path, 'graphs.csv')
    
    if not os.path.exists(graphs_file):
        print(f"错误：{graphs_file} 文件不存在")
        print("请先运行 scn_phase_space_process_v2.py 生成数据")
        return False
    
    # 读取graphs.csv
    print(f"正在读取 {graphs_file}...")
    df = pd.read_csv(graphs_file)
    
    print(f"图数量: {len(df)}")
    print(f"列名: {df.columns.tolist()}")
    
    if 'label' not in df.columns:
        print("错误：没有找到 'label' 列")
        return False
    
    # 检查标签
    labels = df['label'].values
    print(f"原始标签范围: [{labels.min()}, {labels.max()}]")
    print(f"标签类型: {labels.dtype}")
    print(f"唯一标签值: {np.unique(labels)}")
    
    # 检查是否有问题
    has_negative = np.any(labels < 0)
    has_too_large = np.any(labels >= 6)
    has_nan = np.any(pd.isna(labels))
    
    if has_negative:
        print(f"⚠️  发现负数标签: {np.sum(labels < 0)} 个")
    if has_too_large:
        print(f"⚠️  发现过大标签(>=6): {np.sum(labels >= 6)} 个")
    if has_nan:
        print(f"⚠️  发现NaN标签: {np.sum(pd.isna(labels))} 个")
    
    if has_negative or has_too_large or has_nan:
        print("\n开始修复标签...")
        
        # 备份原文件
        backup_file = graphs_file + '.backup'
        df.to_csv(backup_file, index=False)
        print(f"原文件已备份到: {backup_file}")
        
        # 修复标签
        # 1. 处理NaN值
        if has_nan:
            df['label'] = df['label'].fillna(0)
            print("NaN标签已设置为0")
        
        # 2. 限制范围到[0, 5]
        labels_fixed = np.clip(df['label'].values, 0, 5)
        df['label'] = labels_fixed
        
        print(f"修复后标签范围: [{labels_fixed.min()}, {labels_fixed.max()}]")
        print(f"修复后唯一标签值: {np.unique(labels_fixed)}")
        
        # 保存修复后的文件
        df.to_csv(graphs_file, index=False)
        print(f"✓ 标签已修复并保存到: {graphs_file}")
        
        return True
    else:
        print("✓ 标签检查通过，没有发现问题")
        return True


def main():
    """主函数"""
    print("正在进行快速标签修复...\n")
    
    success = quick_fix_labels()
    
    if success:
        print("\n🎉 修复完成！")
        print("现在可以运行以下命令来训练模型：")
        print("python main.py")
    else:
        print("\n❌ 修复失败，请检查数据文件是否存在")
        print("如果数据不存在，请先运行：")
        print("python scn_phase_space_process_v2.py")


if __name__ == "__main__":
    main() 