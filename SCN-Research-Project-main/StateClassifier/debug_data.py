#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据调试脚本

用于检查数据中标签的分布，找出导致CUDA错误的原因
"""

import torch
import numpy as np
import pandas as pd
import os
from utils import get_dataset


def check_data_integrity(data_path='./data'):
    """
    检查数据完整性和标签分布
    
    Parameters
    ----------
    data_path : str
        数据路径
    """
    print("=== 数据完整性检查 ===")
    
    # 检查CSV文件是否存在
    required_files = ['nodes.csv', 'edges.csv', 'graphs.csv']
    for file in required_files:
        filepath = os.path.join(data_path, file)
        if os.path.exists(filepath):
            print(f"✓ {file} 存在")
        else:
            print(f"✗ {file} 不存在")
            return False
    
    # 读取和检查graphs.csv中的标签
    print("\n=== 检查graphs.csv中的标签 ===")
    graphs_df = pd.read_csv(os.path.join(data_path, 'graphs.csv'))
    print(f"图数量: {len(graphs_df)}")
    print(f"标签列: {graphs_df.columns.tolist()}")
    
    if 'label' in graphs_df.columns:
        labels = graphs_df['label'].values
        print(f"标签范围: [{labels.min()}, {labels.max()}]")
        print(f"标签类型: {labels.dtype}")
        print(f"唯一标签值: {np.unique(labels)}")
        print(f"标签分布:")
        for label in np.unique(labels):
            count = np.sum(labels == label)
            print(f"  标签 {label}: {count} 个样本")
        
        # 检查是否有无效标签
        invalid_labels = (labels < 0) | (labels >= 6)
        if np.any(invalid_labels):
            print(f"⚠️  发现 {np.sum(invalid_labels)} 个无效标签！")
            print(f"无效标签值: {labels[invalid_labels]}")
            return False
        else:
            print("✓ 所有标签都在有效范围 [0, 5] 内")
    
    return True


def check_dataloader_labels():
    """
    检查DataLoader中的标签
    """
    print("\n=== 检查DataLoader中的标签 ===")
    
    try:
        # 加载数据集
        train_dataloader, valid_dataloader, test_dataloader = get_dataset('./data')
        
        # 检查训练集标签
        print("检查训练集标签...")
        train_labels = []
        for i, batch in enumerate(train_dataloader):
            y = batch[1]  # 标签
            train_labels.extend(y.numpy())
            if i >= 5:  # 只检查前几个batch
                break
        
        train_labels = np.array(train_labels)
        print(f"训练集标签范围: [{train_labels.min()}, {train_labels.max()}]")
        print(f"训练集唯一标签: {np.unique(train_labels)}")
        
        # 检查是否有无效标签
        invalid_train = (train_labels < 0) | (train_labels >= 6)
        if np.any(invalid_train):
            print(f"⚠️  训练集中发现 {np.sum(invalid_train)} 个无效标签！")
            print(f"无效标签值: {train_labels[invalid_train]}")
            return False
        
        # 检查验证集标签
        print("检查验证集标签...")
        valid_labels = []
        for i, batch in enumerate(valid_dataloader):
            y = batch[1]
            valid_labels.extend(y.numpy())
            if i >= 5:
                break
        
        valid_labels = np.array(valid_labels)
        print(f"验证集标签范围: [{valid_labels.min()}, {valid_labels.max()}]")
        print(f"验证集唯一标签: {np.unique(valid_labels)}")
        
        # 检查测试集标签
        print("检查测试集标签...")
        test_labels = []
        for i, batch in enumerate(test_dataloader):
            y = batch[1]
            test_labels.extend(y.numpy())
            if i >= 5:
                break
        
        test_labels = np.array(test_labels)
        print(f"测试集标签范围: [{test_labels.min()}, {test_labels.max()}]")
        print(f"测试集唯一标签: {np.unique(test_labels)}")
        
        print("✓ DataLoader标签检查完成")
        return True
        
    except Exception as e:
        print(f"✗ DataLoader检查失败: {e}")
        return False


def fix_labels_if_needed(data_path='./data'):
    """
    如果发现标签问题，尝试修复
    """
    print("\n=== 尝试修复标签问题 ===")
    
    graphs_file = os.path.join(data_path, 'graphs.csv')
    if not os.path.exists(graphs_file):
        print("graphs.csv文件不存在，无法修复")
        return False
    
    # 读取graphs.csv
    graphs_df = pd.read_csv(graphs_file)
    
    if 'label' not in graphs_df.columns:
        print("没有找到label列")
        return False
    
    original_labels = graphs_df['label'].values
    print(f"原始标签范围: [{original_labels.min()}, {original_labels.max()}]")
    
    # 修复无效标签
    fixed_labels = np.clip(original_labels, 0, 5)  # 将标签限制在[0, 5]范围内
    
    # 如果有负数，设为0；如果超过5，设为5
    graphs_df['label'] = fixed_labels
    
    # 保存修复后的文件
    backup_file = graphs_file + '.backup'
    graphs_df_original = pd.read_csv(graphs_file)
    graphs_df_original.to_csv(backup_file, index=False)
    print(f"原始文件已备份到: {backup_file}")
    
    graphs_df.to_csv(graphs_file, index=False)
    print(f"修复后的标签范围: [{fixed_labels.min()}, {fixed_labels.max()}]")
    print(f"标签已修复并保存到: {graphs_file}")
    
    return True


def main():
    """主函数"""
    print("开始数据诊断...\n")
    
    # 检查数据完整性
    if not check_data_integrity():
        print("数据完整性检查失败")
        
        # 尝试修复标签
        if fix_labels_if_needed():
            print("标签已修复，请重新运行数据检查")
        else:
            print("标签修复失败")
        return
    
    # 检查DataLoader
    if not check_dataloader_labels():
        print("DataLoader检查失败")
        return
    
    print("\n🎉 数据检查完成，所有标签都在正确范围内！")
    print("现在可以安全运行 main.py")


if __name__ == "__main__":
    main() 