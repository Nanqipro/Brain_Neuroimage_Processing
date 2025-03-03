#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
神经元拓扑结构可视化工具启动脚本
支持参数:
- 网络类型: standard, mst, threshold, top_edges 或 all
- --no-gif: 不生成GIF文件，只生成HTML
- --use-gpu: 使用GPU加速GIF生成
- --no-gpu: 即使有GPU也不使用
"""

import os
import sys
import argparse
from pos_topology_js import main as run_topology

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="神经元拓扑结构分析与可视化工具")
    
    # 网络类型参数
    parser.add_argument('network_type', nargs='?', default='standard',
                        choices=['standard', 'mst', 'threshold', 'top_edges', 'all'],
                        help="要处理的网络类型 (默认: standard)")
    
    # GIF生成选项
    parser.add_argument('--no-gif', action='store_true',
                        help="不生成GIF，只生成HTML输出")
    
    # GPU加速选项
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument('--use-gpu', action='store_true',
                          help="使用GPU加速GIF生成")
    gpu_group.add_argument('--no-gpu', action='store_true',
                          help="即使有GPU也不使用GPU")
    
    return parser.parse_args()

def main():
    """主函数，处理参数并启动拓扑分析"""
    args = parse_arguments()
    
    # 将参数传递给pos_topology_js.py
    sys.argv = sys.argv[:1]  # 清除所有参数
    
    # 添加网络类型参数
    if args.network_type != 'standard':
        sys.argv.append(args.network_type)
    
    # 添加其他参数
    if args.no_gif:
        sys.argv.append('--no-gif')
    if args.use_gpu:
        sys.argv.append('--use-gpu')
    if args.no_gpu:
        sys.argv.append('--no-gpu')
    
    # 运行拓扑分析
    run_topology()

if __name__ == '__main__':
    main() 