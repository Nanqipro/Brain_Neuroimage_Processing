"""
神经元随机位置演示脚本

此脚本演示如何在神经元位置数据文件为空时自动生成随机位置。
展示了两种使用方式：
1. 直接使用空的位置文件路径
2. 使用配置文件中的 'random_positions_demo' 数据集

作者: Assistant
日期: 2025年
"""

import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_neuron_positions, generate_random_neuron_positions
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def demo_random_position_generation():
    """演示随机位置生成功能"""
    print("=" * 60)
    print("神经元随机位置生成演示")
    print("=" * 60)
    
    # === 演示1: 直接生成随机位置 ===
    print("\n=== 演示1: 直接生成随机位置 ===")
    
    # 生成50个神经元的随机位置
    random_positions = generate_random_neuron_positions(num_neurons=50, random_seed=42)
    print(f"生成的随机位置数据形状: {random_positions.shape}")
    print("前5个神经元的位置:")
    print(random_positions.head())
    
    # === 演示2: 通过load_neuron_positions函数使用空路径 ===
    print("\n=== 演示2: 通过load_neuron_positions函数使用空路径 ===")
    
    # 使用空字符串作为路径，会自动生成随机位置
    empty_path_positions = load_neuron_positions("", num_neurons=30, random_seed=123)
    print(f"通过空路径生成的位置数据形状: {empty_path_positions.shape}")
    print("前5个神经元的位置:")
    print(empty_path_positions.head())
    
    # === 演示3: 使用None作为路径 ===
    print("\n=== 演示3: 使用None作为路径 ===")
    
    none_path_positions = load_neuron_positions(None, num_neurons=25, random_seed=456)
    print(f"通过None路径生成的位置数据形状: {none_path_positions.shape}")
    print("前5个神经元的位置:")
    print(none_path_positions.head())
    
    # === 演示4: 可视化随机位置分布 ===
    print("\n=== 演示4: 可视化随机位置分布 ===")
    
    # 生成不同数量的神经元位置进行比较
    neuron_counts = [20, 50, 100]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, num_neurons in enumerate(neuron_counts):
        positions = generate_random_neuron_positions(
            num_neurons=num_neurons, 
            random_seed=42  # 使用相同的种子以便比较
        )
        
        ax = axes[i]
        ax.scatter(positions['x'], positions['y'], alpha=0.7, s=50)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_title(f'{num_neurons} Neurons')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.grid(True, alpha=0.3)
        
        # 添加神经元ID标注（仅对少量神经元）
        if num_neurons <= 20:
            for _, row in positions.iterrows():
                ax.annotate(
                    str(row['NeuronID']), 
                    (row['x'], row['y']), 
                    xytext=(5, 5), 
                    textcoords='offset points',
                    fontsize=8,
                    alpha=0.7
                )
    
    plt.tight_layout()
    plt.suptitle('Random Neuron Position Distributions', y=1.02)
    output_path = '../output_plots/random_positions_demo.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"位置分布图已保存到: {output_path}")
    plt.close()
    
    # === 演示5: 比较不同随机种子的效果 ===
    print("\n=== 演示5: 比较不同随机种子的效果 ===")
    
    seeds = [42, 123, 456]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, seed in enumerate(seeds):
        positions = generate_random_neuron_positions(
            num_neurons=30, 
            random_seed=seed
        )
        
        ax = axes[i]
        ax.scatter(positions['x'], positions['y'], alpha=0.7, s=50)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_title(f'Seed: {seed}')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('Random Positions with Different Seeds (30 Neurons)', y=1.02)
    output_path = '../output_plots/random_seeds_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"种子比较图已保存到: {output_path}")
    plt.close()
    
    return random_positions, empty_path_positions, none_path_positions

def demo_position_statistics():
    """演示位置统计信息"""
    print("\n=== 位置分布统计 ===")
    
    # 生成大量神经元以分析分布特征
    positions = generate_random_neuron_positions(num_neurons=200, random_seed=42)
    
    print(f"总神经元数量: {len(positions)}")
    print(f"X坐标范围: [{positions['x'].min():.3f}, {positions['x'].max():.3f}]")
    print(f"Y坐标范围: [{positions['y'].min():.3f}, {positions['y'].max():.3f}]")
    print(f"X坐标均值: {positions['x'].mean():.3f}")
    print(f"Y坐标均值: {positions['y'].mean():.3f}")
    print(f"X坐标标准差: {positions['x'].std():.3f}")
    print(f"Y坐标标准差: {positions['y'].std():.3f}")
    
    # 计算从中心的平均距离
    center_x, center_y = 0.5, 0.5
    distances = np.sqrt((positions['x'] - center_x)**2 + (positions['y'] - center_y)**2)
    print(f"距离中心的平均距离: {distances.mean():.3f}")
    print(f"距离中心的最大距离: {distances.max():.3f}")

if __name__ == "__main__":
    try:
        # 运行所有演示
        pos1, pos2, pos3 = demo_random_position_generation()
        demo_position_statistics()
        
        print("\n" + "=" * 60)
        print("随机位置生成演示完成!")
        print("=" * 60)
        print("\n使用方法总结:")
        print("1. 在PathConfig中设置position路径为空字符串 ''")
        print("2. 在PathConfig中设置position路径为 None")
        print("3. 直接调用 generate_random_neuron_positions() 函数")
        print("4. 使用 load_neuron_positions('', num_neurons=N) 函数")
        print("\n注意事项:")
        print("- 随机种子确保生成的位置可重复")
        print("- 神经元数量可以通过效应量数据自动确定")
        print("- 生成的位置使用黄金角螺旋算法，分布均匀")
        print("- 所有位置坐标都在 [0, 1] 范围内")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc() 