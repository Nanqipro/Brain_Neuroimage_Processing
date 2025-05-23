#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成测试数据脚本

为了验证模型训练，生成包含多个类别标签的测试数据集
"""

import pandas as pd
import numpy as np
import os


def generate_multi_class_data(data_path='./data', num_classes=6):
    """
    修改现有数据，使其包含多个类别标签
    
    Parameters
    ----------
    data_path : str
        数据目录路径
    num_classes : int
        类别数量，默认6
    """
    print("=== 生成多类别测试数据 ===")
    
    graphs_file = os.path.join(data_path, 'graphs.csv')
    
    if not os.path.exists(graphs_file):
        print(f"错误：{graphs_file} 文件不存在")
        return False
    
    # 读取graphs.csv
    df = pd.read_csv(graphs_file)
    
    # 备份原文件
    backup_file = graphs_file + '.single_class_backup'
    df.to_csv(backup_file, index=False)
    print(f"单类别数据已备份到: {backup_file}")
    
    # 生成多类别标签
    num_graphs = len(df)
    print(f"图数量: {num_graphs}")
    
    # 均匀分布生成标签
    labels_per_class = num_graphs // num_classes
    remainder = num_graphs % num_classes
    
    new_labels = []
    for class_id in range(num_classes):
        # 每个类别分配相同数量的样本
        count = labels_per_class + (1 if class_id < remainder else 0)
        new_labels.extend([class_id] * count)
    
    # 打乱标签顺序
    np.random.seed(42)  # 确保可重现
    np.random.shuffle(new_labels)
    
    # 更新标签
    df['label'] = new_labels
    
    # 显示标签分布
    print("新的标签分布:")
    for class_id in range(num_classes):
        count = np.sum(np.array(new_labels) == class_id)
        percentage = count / num_graphs * 100
        print(f"  类别 {class_id}: {count} 个样本 ({percentage:.1f}%)")
    
    # 保存修改后的文件
    df.to_csv(graphs_file, index=False)
    print(f"✓ 多类别数据已保存到: {graphs_file}")
    
    return True


def main():
    """主函数"""
    print("正在生成多类别测试数据...\n")
    
    success = generate_multi_class_data()
    
    if success:
        print("\n🎉 多类别数据生成完成！")
        print("现在可以运行以下命令来训练模型：")
        print("python main.py")
        print("\n注意：如需恢复原始单类别数据，请使用备份文件：")
        print("cp ./data/graphs.csv.single_class_backup ./data/graphs.csv")
    else:
        print("\n❌ 数据生成失败")


if __name__ == "__main__":
    main() 