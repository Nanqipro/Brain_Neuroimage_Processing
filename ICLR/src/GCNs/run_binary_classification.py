#!/usr/bin/env python3
"""
二分类版本的运行脚本
将8个类别合并为2个：Sleep vs Non-Sleep (Active)
"""

import os
import sys
import argparse

def main():
    """运行二分类实验"""
    parser = argparse.ArgumentParser(description="二分类实验")
    parser.add_argument('--runs', type=int, default=50)
    parser.add_argument('--window_size', type=int, default=20)
    parser.add_argument('--models', nargs='+', default=['gcn', 'gat', 'sage', 'gin'])
    args = parser.parse_args()
    
    # 创建类别映射文件
    label_mapping = {
        'Sleep': 'Sleep',
        'Wake': 'Active',
        'Move': 'Active',
        'Drink': 'Active',
        'zone': 'Active',
        'Scratch': 'Active',
        'Groom': 'Active',
        'Active': 'Active'
    }
    
    # 保存映射
    import json
    with open('label_mapping.json', 'w') as f:
        json.dump(label_mapping, f)
    
    print("使用二分类模式：Sleep vs Active (所有非Sleep行为)")
    print(f"原8类别映射：{label_mapping}")
    
    # 运行批量实验
    cmd = f"python batch_experiments.py --runs {args.runs} --window_sizes {args.window_size} --models {' '.join(args.models)}"
    os.system(cmd)

if __name__ == "__main__":
    main()
