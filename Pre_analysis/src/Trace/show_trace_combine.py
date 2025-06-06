import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Any

# 确保输出目录存在
output_dir = "../../graph/heatmap-trace-combine/"
os.makedirs(output_dir, exist_ok=True)

# 加载神经元对应表
neuron_map = pd.read_excel("../../datasets/神经元对应表2979.xlsx", sheet_name="Sheet1")

# 加载 Day3、Day6 和 Day9 的钙离子浓度数据
day3_data = pd.read_excel("../../datasets/Day3_with_behavior_labels_filled.xlsx")
day6_data = pd.read_excel("../../datasets/Day6_with_behavior_labels_filled.xlsx")
day9_data = pd.read_excel("../../datasets/Day9_with_behavior_labels_filled.xlsx")

def find_common_neurons() -> List[Dict[str, Any]]:
    """
    筛选在所有三个数据文件中都存在的神经元
    
    Returns
    -------
    List[Dict[str, Any]]
        包含所有文件中都存在的神经元信息列表
    """
    common_neurons = []
    
    for idx, row in neuron_map.iterrows():
        day3_neuron = row['Day3_with_behavior_labels_filled']
        day6_neuron = row['Day6_with_behavior_labels_filled']
        day9_neuron = row['Day9_with_behavior_labels_filled']
        
        # 检查神经元是否在所有三个数据文件中都存在
        if (day3_neuron in day3_data.columns and 
            day6_neuron in day6_data.columns and 
            day9_neuron in day9_data.columns and
            day3_neuron != 'null' and 
            day6_neuron != 'null' and 
            day9_neuron != 'null'):
            
            common_neurons.append({
                'day3_id': day3_neuron,
                'day6_id': day6_neuron, 
                'day9_id': day9_neuron,
                'index': idx
            })
    
    return common_neurons

def plot_traces_with_same_colors(common_neurons: List[Dict[str, Any]]) -> None:
    """
    绘制trace图，每个神经元在三天中使用相同颜色
    
    Parameters
    ----------
    common_neurons : List[Dict[str, Any]]
        包含共同神经元信息的列表
    """
    if not common_neurons:
        print("未找到在所有文件中都存在的神经元")
        return
    
    # 生成颜色映射
    colors = plt.cm.tab10(np.linspace(0, 1, len(common_neurons)))
    
    # 创建图形
    fig, axes = plt.subplots(len(common_neurons), 1, figsize=(14, 3*len(common_neurons)))
    if len(common_neurons) == 1:
        axes = [axes]
    
    for idx, neuron_info in enumerate(common_neurons):
        ax = axes[idx]
        color = colors[idx]
        
        # 获取神经元ID
        day3_id = neuron_info['day3_id']
        day6_id = neuron_info['day6_id'] 
        day9_id = neuron_info['day9_id']
        
        # 绘制三天的trace，使用相同颜色但不同线型
        ax.plot(day3_data['stamp'], day3_data[day3_id], 
                color=color, linestyle='-', linewidth=1.5, label="Day3", alpha=0.8)
        ax.plot(day6_data['stamp'], day6_data[day6_id], 
                color=color, linestyle='--', linewidth=1.5, label="Day6", alpha=0.8)
        ax.plot(day9_data['stamp'], day9_data[day9_id], 
                color=color, linestyle='-.', linewidth=1.5, label="Day9", alpha=0.8)
        
        # 设置图表属性
        ax.set_title(f"Neuron {day3_id} trace across three days", fontsize=12)
        ax.set_xlabel("Time stamp")
        ax.set_ylabel("Ca2+ concentration")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, "common_neurons_traces.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Trace图已保存到: {output_path}")
    plt.show()

def plot_combined_heatmap(common_neurons: List[Dict[str, Any]]) -> None:
    """
    绘制只包含共同神经元的热图
    
    Parameters
    ----------
    common_neurons : List[Dict[str, Any]]
        包含共同神经元信息的列表
    """
    if not common_neurons:
        return
        
    # 构建对齐数据
    aligned_data = {}
    for neuron_info in common_neurons:
        day3_id = neuron_info['day3_id']
        day6_id = neuron_info['day6_id']
        day9_id = neuron_info['day9_id']
        
        # 提取神经元数据
        day3_values = day3_data[day3_id]
        day6_values = day6_data[day6_id]
        day9_values = day9_data[day9_id]
        
        # 将数据存储到字典中
        aligned_data[day3_id] = pd.DataFrame({
            'Day3': day3_values.values,
            'Day6': day6_values.values,
            'Day9': day9_values.values
        })
    
    # 组合数据为DataFrame
    final_aligned_df = pd.concat(aligned_data, axis=1)
    
    # 生成热图
    plt.figure(figsize=(12, 8))
    sns.heatmap(final_aligned_df.T, cmap="viridis", cbar=True)
    plt.title("Common neurons calcium concentration heatmap")
    plt.xlabel("Time stamp")
    plt.ylabel("Neurons across days")
    
    # 保存热图
    heatmap_path = os.path.join(output_dir, "common_neurons_heatmap.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"热图已保存到: {heatmap_path}")
    plt.show()

# 主执行流程
if __name__ == "__main__":
    # 找到共同的神经元
    common_neurons = find_common_neurons()
    print(f"找到 {len(common_neurons)} 个在所有文件中都存在的神经元")
    
    if common_neurons:
        # 打印找到的神经元信息
        print("共同神经元列表:")
        for i, neuron in enumerate(common_neurons):
            print(f"  {i+1}. Day3: {neuron['day3_id']}, Day6: {neuron['day6_id']}, Day9: {neuron['day9_id']}")
        
        # 绘制trace图
        plot_traces_with_same_colors(common_neurons)
        
        # 绘制热图
        plot_combined_heatmap(common_neurons)
    else:
        print("警告：未找到在所有三个文件中都存在的神经元")