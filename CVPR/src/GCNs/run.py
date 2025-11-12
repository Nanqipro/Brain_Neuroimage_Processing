#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
通用图分类模型训练脚本

支持三种数据源：
1. 原始CSV数据（神经元数据）- 需要构建图
2. TUDataset标准数据集 - 已有图结构
3. OGB数据集 - Open Graph Benchmark大规模图基准

使用方法：
    # 使用TUDataset（推荐用于标准基准测试）
    python run.py --data_source tudataset --dataset MUTAG --model gcn
    
    # 使用OGB数据集（大规模图基准测试）
    python run.py --data_source ogb --dataset ogbg-molhiv --model gin
    
    # 使用原始CSV数据（用于神经元数据）
    python run.py --data_source csv --dataset dataset/processed3.csv --model gcn
"""

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import os
import datetime
import time
import json
import argparse
import psutil
import GPUtil
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

# # -----------------------------------------------------------------
# # 解决 PyTorch 2.6+ 安全加载 OGB/PyG 数据集的问题
# # (torch.load weights_only=True)
# # -----------------------------------------------------------------
# try:
#     # 导入 torch 和所有需要的 torch_geometric 模块
#     import torch
#     import torch_geometric.data.data 
#     import torch_geometric.data.storage 

#     # 明确告诉 PyTorch 信任这些来自 torch_geometric 的类
#     torch.serialization.add_safe_globals([
#         torch_geometric.data.data.DataEdgeAttr,
#         torch_geometric.data.data.DataTensorAttr,
#         torch_geometric.data.storage.GlobalStorage
#     ])
# except ImportError:
#     # 如果环境不完整，先跳过，后续代码会正常报错
#     pass
# # -----------------------------------------------------------------

# 导入模型和工具函数
from model import (
    ImprovedGCN, PureGCN, PureGAT, PureGraphSAGE,
    GIN, ChebNet, EdgeConvNet, GraphUNet,
    PNA, GATv2, DeeperGCN
)
from train import train_model, evaluate_model, plot_confusion_matrix, plot_training_metrics
from process import (load_data, oversample_data, compute_correlation_matrix, 
                     create_pyg_dataset, visualize_graph, load_custom_graph_dataset)

# ============================================================================
# 资源监控函数
# ============================================================================

class ResourceMonitor:
    """监控CPU内存和GPU显存使用"""
    
    def __init__(self, device):
        self.device = device
        self.process = psutil.Process()
        self.gpu_available = torch.cuda.is_available()
        
    def get_memory_usage(self):
        """获取当前内存使用情况（MB）"""
        mem_info = self.process.memory_info()
        return mem_info.rss / 1024 / 1024  # 转换为MB
    
    def get_gpu_memory_usage(self):
        """获取当前GPU显存使用情况（MB）"""
        if not self.gpu_available:
            return 0.0
        
        if torch.cuda.is_available():
            # PyTorch方式获取显存
            allocated = torch.cuda.memory_allocated(self.device) / 1024 / 1024
            reserved = torch.cuda.memory_reserved(self.device) / 1024 / 1024
            return {
                'allocated': allocated,  # 实际分配的显存
                'reserved': reserved,     # 预留的显存
                'max_allocated': torch.cuda.max_memory_allocated(self.device) / 1024 / 1024
            }
        return {'allocated': 0.0, 'reserved': 0.0, 'max_allocated': 0.0}
    
    def get_snapshot(self):
        """获取当前资源使用快照"""
        gpu_mem = self.get_gpu_memory_usage()
        return {
            'cpu_memory_mb': self.get_memory_usage(),
            'gpu_memory_allocated_mb': gpu_mem.get('allocated', 0.0) if isinstance(gpu_mem, dict) else 0.0,
            'gpu_memory_reserved_mb': gpu_mem.get('reserved', 0.0) if isinstance(gpu_mem, dict) else 0.0,
            'gpu_memory_max_allocated_mb': gpu_mem.get('max_allocated', 0.0) if isinstance(gpu_mem, dict) else 0.0
        }


# 添加绘制时间曲线的函数
def plot_epoch_time(epoch_times, cumulative_times, result_dir='result'):
    """绘制累积执行时间曲线
    
    Args:
        epoch_times: 每个epoch的单独执行时间列表（秒）
        cumulative_times: 从训练开始的累积时间列表（秒）
        result_dir: 结果保存目录
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 6))
    epochs = list(range(1, len(cumulative_times) + 1))
    
    # 绘制累积时间曲线
    plt.plot(epochs, cumulative_times, 'b-o', linewidth=2, markersize=6, label='Cumulative Time')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Cumulative Time (seconds)', fontsize=12)
    plt.title('Cumulative Training Time from Start', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # 添加最终累积时间标注
    final_time = cumulative_times[-1] if cumulative_times else 0
    plt.axhline(y=final_time, color='r', linestyle='--', linewidth=2, alpha=0.5,
                label=f'Total Time: {final_time:.3f}s')
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f'{result_dir}/epoch_time.png', dpi=300)
    plt.close()
    
    # 统计信息
    avg_epoch_time = np.mean(epoch_times) if epoch_times else 0
    print(f"\n时间统计:")
    print(f"  - 累积总时间: {final_time:.3f}秒")
    print(f"  - 平均每个epoch时间: {avg_epoch_time:.3f}秒")
    print(f"  - 最快epoch时间: {min(epoch_times):.3f}秒" if epoch_times else "  - 最快epoch时间: 0秒")
    print(f"  - 最慢epoch时间: {max(epoch_times):.3f}秒" if epoch_times else "  - 最慢epoch时间: 0秒")
    print(f"  - epoch时间标准差: {np.std(epoch_times):.3f}秒" if epoch_times else "  - epoch时间标准差: 0秒")


def save_training_history_csv(history, result_dir='result'):
    """保存训练历史到CSV文件
    
    Args:
        history: 训练历史字典
        result_dir: 结果保存目录
    """
    # 构建DataFrame
    num_epochs = len(history['epoch_times'])
    
    data = {
        'epoch': list(range(1, num_epochs + 1)),
        'train_time': history['train_times'],  # 每epoch训练时间
        'val_time': history['val_times'],  # 每epoch验证时间
        'epoch_time': history['epoch_times'],  # 每epoch总时间
        'cumulative_train_time': history['cumulative_train_times'],  # 累积训练时间
        'train_loss': history['train']['loss'],
        'train_accuracy': history['train']['accuracy'],
        'train_precision': history['train']['precision'],
        'train_recall': history['train']['recall'],
        'train_f1': history['train']['f1'],
        'val_accuracy': history['val']['accuracy'],
        'val_precision': history['val']['precision'],
        'val_recall': history['val']['recall'],
        'val_f1': history['val']['f1']
    }
    
    df = pd.DataFrame(data)
    
    # 保存到CSV
    csv_path = f'{result_dir}/training_history.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8')
    
    print(f"\n训练历史已保存到: {csv_path}")
    print(f"  - 包含 {num_epochs} 个epoch的详细数据")
    print(f"  - 可直接用于绘图和分析")


def save_resource_usage_log(resource_log, num_train_graphs, total_training_time, avg_training_time, result_dir='result'):
    """保存资源使用日志到单独的文件
    
    Args:
        resource_log: 资源使用记录字典
        num_train_graphs: 训练集图的数量
        total_training_time: 总训练时间（仅训练集，不包括验证集）
        avg_training_time: 平均每epoch的训练时间（仅训练集）
        result_dir: 结果保存目录
    """
    log_file = f'{result_dir}/resource_usage.txt'
    
    # 计算统计信息
    avg_cpu_mem = np.mean(resource_log['cpu_memory'])
    max_cpu_mem = np.max(resource_log['cpu_memory'])
    avg_gpu_mem = np.mean(resource_log['gpu_memory_allocated'])
    max_gpu_mem = np.max(resource_log['gpu_memory_max_allocated'])
    
    # 计算每个图的平均执行时间（仅基于训练集）
    time_per_graph = total_training_time / num_train_graphs if num_train_graphs > 0 else 0
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("资源使用统计报告（基于训练集时间）\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("时间统计（仅训练集，不含验证集）\n")
        f.write("-" * 80 + "\n")
        f.write(f"训练集图数量: {num_train_graphs}\n")
        f.write(f"总训练时间: {total_training_time:.3f} 秒\n")
        f.write(f"平均每epoch训练时间: {avg_training_time:.3f} 秒\n")
        f.write(f"每个图的平均执行时间: {time_per_graph:.6f} 秒/图\n")
        f.write(f"每个图的平均执行时间: {time_per_graph*1000:.3f} 毫秒/图\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("内存使用统计 (CPU)\n")
        f.write("-" * 80 + "\n")
        f.write(f"平均内存使用: {avg_cpu_mem:.2f} MB\n")
        f.write(f"峰值内存使用: {max_cpu_mem:.2f} MB\n")
        f.write(f"内存使用范围: {np.min(resource_log['cpu_memory']):.2f} - {max_cpu_mem:.2f} MB\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("显存使用统计 (GPU)\n")
        f.write("-" * 80 + "\n")
        f.write(f"平均显存使用 (已分配): {avg_gpu_mem:.2f} MB\n")
        f.write(f"峰值显存使用: {max_gpu_mem:.2f} MB\n")
        f.write(f"平均显存预留: {np.mean(resource_log['gpu_memory_reserved']):.2f} MB\n")
        f.write(f"峰值显存预留: {np.max(resource_log['gpu_memory_reserved']):.2f} MB\n\n")
        
        f.write("=" * 80 + "\n")
    
    # 同时保存到CSV供后续分析
    csv_file = f'{result_dir}/resource_usage.csv'
    df = pd.DataFrame({
        'epoch': list(range(1, len(resource_log['cpu_memory']) + 1)),
        'cpu_memory_mb': resource_log['cpu_memory'],
        'gpu_memory_allocated_mb': resource_log['gpu_memory_allocated'],
        'gpu_memory_reserved_mb': resource_log['gpu_memory_reserved'],
        'gpu_memory_max_allocated_mb': resource_log['gpu_memory_max_allocated']
    })
    df.to_csv(csv_file, index=False, encoding='utf-8')
    
    print(f"\n资源使用日志已保存:")
    print(f"  - 文本报告: {log_file}")
    print(f"  - CSV数据: {csv_file}")
    print(f"  - 每个图平均执行时间: {time_per_graph*1000:.3f} 毫秒/图")
    print(f"  - 峰值内存: {max_cpu_mem:.2f} MB")
    print(f"  - 峰值显存: {max_gpu_mem:.2f} MB")


# 模型字典
MODEL_DICT = {
    # 原有基础模型
    'hybrid': ImprovedGCN,
    'gcn': PureGCN,
    'gat': PureGAT,
    'sage': PureGraphSAGE,
    # 先进模型 (2016-2019)
    'gin': GIN,              # Graph Isomorphism Network (ICLR 2019) - 理论最强
    'chebnet': ChebNet,      # Chebyshev GCN (NIPS 2016) - 高效
    'edgeconv': EdgeConvNet, # Dynamic Graph CNN (TOG 2019) - 动态图
    'gunet': GraphUNet,      # Graph U-Net (ICML 2019) - 多尺度
    # 最新模型 (2020-2022)
    'pna': PNA,              # Principal Neighbourhood Aggregation (NeurIPS 2020) - 多聚合器
    'gatv2': GATv2,          # Graph Attention v2 (ICLR 2022) - 改进注意力
    'deepergcn': DeeperGCN,  # DeeperGCN (ICLR 2020) - 深层网络
}


# ============================================================================
# 数据加载函数
# ============================================================================

def load_csv_data(data_path, seed, train_ratio=0.8, val_ratio=0.1):
    """
    从CSV文件加载神经元数据并构建图
    
    Args:
        data_path: CSV文件路径
        seed: 随机种子
        train_ratio: 训练集比例
        val_ratio: 验证集比例
    
    Returns:
        train_loader, val_loader, test_loader, num_features, num_classes, class_weights
    """
    print(f"\n加载CSV数据: {data_path}")
    
    # 加载数据
    features, labels, class_weights, class_names = load_data(data_path)
    
    # SMOTE过采样
    features_resampled, labels_resampled = oversample_data(features, labels, ramdom_state=seed)
    
    # 划分数据集
    X_train, X_temp, y_train, y_temp = train_test_split(
        features_resampled, labels_resampled, 
        test_size=(1-train_ratio), 
        random_state=seed, 
        stratify=labels_resampled
    )
    
    val_size = val_ratio / (1 - train_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, 
        test_size=(1-val_size), 
        random_state=seed, 
        stratify=y_temp
    )
    
    print(f"\n数据集划分:")
    print(f"  - 训练集: {len(X_train)} 样本")
    print(f"  - 验证集: {len(X_val)} 样本")
    print(f"  - 测试集: {len(X_test)} 样本")
    
    # 计算相关性矩阵并构建图
    print("\n构建图结构...")
    correlation_matrix = compute_correlation_matrix(X_train)
    
    train_data_list = create_pyg_dataset(X_train, y_train, correlation_matrix)
    val_data_list = create_pyg_dataset(X_val, y_val, correlation_matrix)
    test_data_list = create_pyg_dataset(X_test, y_test, correlation_matrix)
    
    # 创建DataLoader
    train_loader = DataLoader(train_data_list, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data_list, batch_size=32)
    test_loader = DataLoader(test_data_list, batch_size=32)
    
    num_features = 1  # CSV数据每个节点特征是标量
    num_classes = len(np.unique(labels))
    
    return train_loader, val_loader, test_loader, num_features, num_classes, class_weights, class_names


def load_tudataset_data(dataset_name, data_root, seed, train_ratio=0.8, val_ratio=0.1, batch_size=32):
    """
    加载TUDataset标准数据集
    
    Args:
        dataset_name: 数据集名称，如'MUTAG', 'PROTEINS'
        data_root: 数据集根目录
        seed: 随机种子
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        batch_size: 批大小
    
    Returns:
        train_loader, val_loader, test_loader, num_features, num_classes, None, None
    """
    from torch_geometric.datasets import TUDataset
    
    print(f"\n加载TUDataset: {dataset_name}")
    
    # 加载数据集
    dataset = TUDataset(root=data_root, name=dataset_name)
    
    print(f"\n数据集信息:")
    print(f"  - 图数量: {len(dataset)}")
    print(f"  - 特征维度: {dataset.num_features}")
    print(f"  - 类别数: {dataset.num_classes}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"  - 示例图节点数: {sample.num_nodes}")
        print(f"  - 示例图边数: {sample.num_edges}")
    
    # 划分数据集
    num_graphs = len(dataset)
    indices = list(range(num_graphs))
    
    # 分层抽样
    if hasattr(dataset[0], 'y'):
        labels = [data.y.item() for data in dataset]
        
        train_idx, temp_idx = train_test_split(
            indices, train_size=train_ratio, 
            random_state=seed, stratify=labels
        )
        
        temp_labels = [labels[i] for i in temp_idx]
        val_size = val_ratio / (1 - train_ratio)
        
        val_idx, test_idx = train_test_split(
            temp_idx, train_size=val_size,
            random_state=seed, stratify=temp_labels
        )
    else:
        np.random.seed(seed)
        np.random.shuffle(indices)
        train_size = int(num_graphs * train_ratio)
        val_size = int(num_graphs * val_ratio)
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]
    
    # 创建子数据集
    train_dataset = [dataset[i] for i in train_idx]
    val_dataset = [dataset[i] for i in val_idx]
    test_dataset = [dataset[i] for i in test_idx]
    
    print(f"\n数据集划分:")
    print(f"  - 训练集: {len(train_dataset)} 图")
    print(f"  - 验证集: {len(val_dataset)} 图")
    print(f"  - 测试集: {len(test_dataset)} 图")
    
    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader, dataset.num_features, dataset.num_classes, None, None


def load_ogb_data(dataset_name, data_root, batch_size=32):
    """
    加载OGB (Open Graph Benchmark) 数据集
    
    Args:
        dataset_name: OGB数据集名称，如'ogbg-molhiv', 'ogbg-molpcba'等
        data_root: 数据集根目录
        batch_size: 批大小
    
    Returns:
        train_loader, val_loader, test_loader, num_features, num_classes, None, None
    """
    from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
    
    print(f"\n加载OGB数据集: {dataset_name}")
    
    # 加载数据集
    dataset = PygGraphPropPredDataset(name=dataset_name, root=data_root)
    
    # 获取预定义的划分
    split_idx = dataset.get_idx_split()
    
    print(f"\n数据集信息:")
    print(f"  - 图数量: {len(dataset)}")
    print(f"  - 特征维度: {dataset.num_features}")
    print(f"  - 任务类型: {dataset.task_type}")
    print(f"  - 评估指标: {dataset.eval_metric}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"  - 示例图节点数: {sample.num_nodes}")
        print(f"  - 示例图边数: {sample.num_edges}")
    
    # 检查是否有节点特征
    has_node_features = dataset.num_features > 0
    
    # 创建子数据集并转换特征类型
    # OGB数据集的节点特征通常是int64类型，需要转换为float32以兼容所有GNN模型
    def convert_data_to_float(data):
        """将图数据的节点特征转换为float32类型，或为无特征数据集生成特征"""
        # 检查是否需要生成特征
        needs_features = (data.x is None or 
                         (hasattr(data.x, 'shape') and (len(data.x.shape) == 0 or data.x.shape[1] == 0)))
        
        if needs_features:
            # 如果没有节点特征，使用节点度数作为特征
            from torch_geometric.utils import degree
            edge_index = data.edge_index
            num_nodes = data.num_nodes
            
            # 计算节点度数（入度 + 出度）
            row, col = edge_index
            deg = degree(row, num_nodes, dtype=torch.float) + degree(col, num_nodes, dtype=torch.float)
            
            # 将度数转换为特征 (num_nodes, 1)
            data.x = deg.view(-1, 1).float()
        elif data.x is not None and data.x.dtype != torch.float32:
            # 转换为 float32
            data.x = data.x.float()
        
        return data
    
    train_dataset = [convert_data_to_float(dataset[i]) for i in split_idx['train']]
    val_dataset = [convert_data_to_float(dataset[i]) for i in split_idx['valid']]
    test_dataset = [convert_data_to_float(dataset[i]) for i in split_idx['test']]
    
    print(f"\n数据集划分 (预定义):")
    print(f"  - 训练集: {len(train_dataset)} 图")
    print(f"  - 验证集: {len(val_dataset)} 图")
    print(f"  - 测试集: {len(test_dataset)} 图")
    if has_node_features:
        print(f"  - 节点特征已转换为float32类型")
    else:
        print(f"  - ⚠️  原数据集无节点特征，已自动生成度数特征 (维度=1)")
        print(f"  - 使用节点度数 (入度+出度) 作为节点特征")
    
    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # OGB数据集的类别数处理
    # 对于二分类任务，num_tasks为1，但我们需要2个输出类别
    if dataset.task_type == 'binary classification':
        num_classes = 2
    elif dataset.task_type == 'multiclass classification':
        num_classes = dataset.num_classes
    else:
        # 对于回归或多标签任务
        num_classes = dataset.num_tasks
    
    # 更新特征维度（如果生成了度数特征）
    actual_num_features = dataset.num_features if has_node_features else 1
    
    return train_loader, val_loader, test_loader, actual_num_features, num_classes, None, None


def load_custom_data(data_dir, seed, train_ratio=0.8, val_ratio=0.1, batch_size=32, 
                     feature_dim=16, num_classes=2, use_degree_feature=False):
    """
    加载自定义图结构数据集（如random数据集）
    
    Args:
        data_dir: 数据集目录路径
        seed: 随机种子
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        batch_size: 批大小
        feature_dim: 随机节点特征维度
        num_classes: 类别数
        use_degree_feature: 是否使用节点度数作为特征
    
    Returns:
        train_loader, val_loader, test_loader, num_features, num_classes, None, None
    """
    print(f"\n加载自定义图数据集: {data_dir}")
    
    # 加载数据集
    data_list, num_features, actual_num_classes = load_custom_graph_dataset(
        data_dir, 
        feature_dim=feature_dim,
        num_classes=num_classes,
        use_degree_feature=use_degree_feature
    )
    
    # 划分数据集
    num_graphs = len(data_list)
    indices = list(range(num_graphs))
    
    # 分层抽样
    labels = [data.y.item() for data in data_list]
    
    train_idx, temp_idx = train_test_split(
        indices, train_size=train_ratio, 
        random_state=seed, stratify=labels
    )
    
    temp_labels = [labels[i] for i in temp_idx]
    val_size = val_ratio / (1 - train_ratio)
    
    val_idx, test_idx = train_test_split(
        temp_idx, train_size=val_size,
        random_state=seed, stratify=temp_labels
    )
    
    # 创建子数据集
    train_dataset = [data_list[i] for i in train_idx]
    val_dataset = [data_list[i] for i in val_idx]
    test_dataset = [data_list[i] for i in test_idx]
    
    print(f"\n数据集划分:")
    print(f"  - 训练集: {len(train_dataset)} 图")
    print(f"  - 验证集: {len(val_dataset)} 图")
    print(f"  - 测试集: {len(test_dataset)} 图")
    
    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader, num_features, actual_num_classes, None, None


# ============================================================================
# 训练函数
# ============================================================================

def setup_result_directory(model_name, data_source, dataset_name):
    """创建结果保存目录"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if data_source == 'csv':
        base_dir = f"result/csv/{model_name}"
    elif data_source == 'ogb':
        dataset_simple = dataset_name.split('/')[-1]
        base_dir = f"result/ogb/{dataset_simple}/{model_name}"
    elif data_source == 'custom':
        dataset_simple = dataset_name.split('/')[-1]
        base_dir = f"result/custom/{dataset_simple}/{model_name}"
    else:  # tudataset
        dataset_simple = dataset_name.split('/')[-1].replace('.csv', '')
        base_dir = f"result/tudataset/{dataset_simple}/{model_name}"
    
    os.makedirs(base_dir, exist_ok=True)
    result_dir = f"{base_dir}/{timestamp}"
    os.makedirs(result_dir, exist_ok=True)
    
    return result_dir


def run_experiment(args):
    """运行完整实验"""
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 设置设备
    if torch.cuda.is_available():
        if args.gpu_id >= 0:
            device = torch.device(f'cuda:{args.gpu_id}')
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print("=" * 80)
    print("通用图分类模型训练")
    print("=" * 80)
    print(f"数据源: {args.data_source}")
    print(f"数据集: {args.dataset}")
    print(f"模型: {args.model}")
    print(f"设备: {device}")
    print("=" * 80)
    
    # 记录开始时间
    begin_time = time.time()
    
    # 初始化资源监控器
    resource_monitor = ResourceMonitor(device)
    
    # 根据数据源加载数据
    if args.data_source == 'csv':
        train_loader, val_loader, test_loader, num_features, num_classes, class_weights, class_names = \
            load_csv_data(args.dataset, args.seed, args.train_ratio, args.val_ratio)
        class_weights = class_weights.to(device) if class_weights is not None else None
    elif args.data_source == 'tudataset':
        train_loader, val_loader, test_loader, num_features, num_classes, _, _ = \
            load_tudataset_data(
                args.dataset, args.data_root, args.seed,
                args.train_ratio, args.val_ratio, args.batch_size
            )
        class_weights = None
        class_names = [str(i) for i in range(num_classes)]
    elif args.data_source == 'ogb':
        train_loader, val_loader, test_loader, num_features, num_classes, _, _ = \
            load_ogb_data(
                args.dataset, args.ogb_root, args.batch_size
            )
        class_weights = None
        class_names = [str(i) for i in range(num_classes)]
    else:  # custom
        train_loader, val_loader, test_loader, num_features, num_classes, _, _ = \
            load_custom_data(
                args.dataset, args.seed, args.train_ratio, args.val_ratio, args.batch_size,
                args.feature_dim, args.num_classes, args.use_degree_feature
            )
        class_weights = None
        class_names = [str(i) for i in range(num_classes)]
    
    # 统计训练集图的数量
    num_train_graphs = len(train_loader.dataset)
    num_val_graphs = len(val_loader.dataset)
    num_test_graphs = len(test_loader.dataset)
    total_graphs = num_train_graphs + num_val_graphs + num_test_graphs
    
    print(f"\n图数量统计:")
    print(f"  - 训练集图数: {num_train_graphs}")
    print(f"  - 验证集图数: {num_val_graphs}")
    print(f"  - 测试集图数: {num_test_graphs}")
    print(f"  - 总图数: {total_graphs}")
    
    # 创建模型
    model_class = MODEL_DICT[args.model]
    model = model_class(
        num_features=num_features,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        dropout=args.dropout
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型信息:")
    print(f"  - 模型类型: {args.model}")
    print(f"  - 参数量: {total_params}")
    print(f"  - 隐藏层维度: {args.hidden_dim}")
    print(f"  - Dropout: {args.dropout}")
    
    # 在模型创建后重置GPU显存统计（此时CUDA context已完全初始化）
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
        print(f"  - GPU显存统计已重置")
    
    # 优化器和学习率调度器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=20
    )
    
    # 训练历史
    history = {
        'train': {'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
        'val': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
        'train_times': [],  # 每个epoch的训练时间（仅训练集）
        'val_times': [],  # 每个epoch的验证时间（仅验证集）
        'epoch_times': [],  # 每个epoch的总时间（训练+验证）
        'cumulative_train_times': []  # 累积训练时间（仅训练集）
    }
    
    # 资源使用记录
    resource_log = {
        'cpu_memory': [],
        'gpu_memory_allocated': [],
        'gpu_memory_reserved': [],
        'gpu_memory_max_allocated': []
    }
    
    best_val_f1 = 0
    best_epoch = 0
    patience_counter = 0
    
    print(f"\n开始训练 (最多 {args.epochs} 轮)...")
    print("=" * 80)
    
    # 记录训练开始时间（用于计算累积时间）
    training_start_time = time.time()
    
    # 训练循环
    for epoch in range(1, args.epochs + 1):
        # 记录epoch开始时间
        epoch_start_time = time.time()
        
        # 训练（只记录训练集时间）
        train_start_time = time.time()
        train_metrics = train_model(model, train_loader, optimizer, device, class_weights)
        train_end_time = time.time()
        train_time = train_end_time - train_start_time
        history['train_times'].append(train_time)
        
        # 验证（记录验证集时间，但不用于 time_per_graph 计算）
        val_start_time = time.time()
        val_metrics = evaluate_model(model, val_loader, device)
        val_end_time = time.time()
        val_time = val_end_time - val_start_time
        history['val_times'].append(val_time)
        
        # 记录epoch总时间（训练+验证）
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        history['epoch_times'].append(epoch_time)
        
        # 记录累积训练时间（仅训练集，从训练开始到当前epoch的训练结束）
        cumulative_train_time = sum(history['train_times'])
        history['cumulative_train_times'].append(cumulative_train_time)
        
        # 记录资源使用情况
        resource_snapshot = resource_monitor.get_snapshot()
        resource_log['cpu_memory'].append(resource_snapshot['cpu_memory_mb'])
        resource_log['gpu_memory_allocated'].append(resource_snapshot['gpu_memory_allocated_mb'])
        resource_log['gpu_memory_reserved'].append(resource_snapshot['gpu_memory_reserved_mb'])
        resource_log['gpu_memory_max_allocated'].append(resource_snapshot['gpu_memory_max_allocated_mb'])
        
        # 学习率调度
        scheduler.step(val_metrics['f1'])
        
        # 记录历史
        for metric in ['loss', 'accuracy', 'precision', 'recall', 'f1']:
            if metric in train_metrics:
                history['train'][metric].append(train_metrics[metric])
        
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            if metric in val_metrics:
                history['val'][metric].append(val_metrics[metric])
        
        # 保存最佳模型
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_epoch = epoch
            patience_counter = 0
            
            if args.save_model:
                result_dir = setup_result_directory(args.model, args.data_source, args.dataset)
                torch.save(model.state_dict(), f'{result_dir}/best_model.pth')
        else:
            patience_counter += 1
        
        # 打印进度
        if epoch % args.print_every == 0:
            print(f"Epoch {epoch:03d}: Loss={train_metrics['loss']:.4f}, "
                  f"Train F1={train_metrics['f1']:.4f}, Val F1={val_metrics['f1']:.4f}, "
                  f"Val Acc={val_metrics['accuracy']:.4f}, "
                  f"Train Time={train_time:.3f}s, Val Time={val_time:.3f}s")
        
        # Early stopping (已禁用，训练固定epoch数)
        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    print("=" * 80)
    
    # 计算到达早停（或最大epoch）时的实际训练指标（仅使用训练集时间）
    actual_epochs = len(history['train_times'])
    total_training_time = history['cumulative_train_times'][-1] if history['cumulative_train_times'] else 0  # 仅训练集时间
    avg_training_time = total_training_time / actual_epochs if actual_epochs > 0 else 0  # 平均每epoch的训练时间
    
    # 计算验证集总时间（用于参考，但不用于 time_per_graph 计算）
    total_val_time = sum(history['val_times']) if history['val_times'] else 0
    total_time_with_val = total_training_time + total_val_time  # 训练+验证总时间
    
    # 计算每个图的平均执行时间（仅基于训练集）
    time_per_graph = total_training_time / num_train_graphs if num_train_graphs > 0 else 0
    
    # 计算每个图在单个epoch中的平均执行时间（另一种计算方式）
    time_per_graph_per_epoch = avg_training_time / num_train_graphs if num_train_graphs > 0 else 0
    
    # 测试（记录推理时间）
    print(f"\n在测试集上评估...")
    test_start_time = time.time()
    test_metrics = evaluate_model(model, test_loader, device)
    test_end_time = time.time()
    test_inference_time = test_end_time - test_start_time
    
    # 计算脚本总运行时间
    total_elapsed_time = time.time() - begin_time
    
    # 打印结果
    print(f"\n{'=' * 80}")
    print(f"实验结果:")
    print(f"{'=' * 80}")
    print(f"最佳验证F1 (Epoch {best_epoch}): {best_val_f1:.4f}")
    print(f"实际训练轮数: {actual_epochs} epochs")
    print(f"\n测试集性能:")
    print(f"  - 准确率: {test_metrics['accuracy']:.4f}")
    print(f"  - 精确率: {test_metrics['precision']:.4f}")
    print(f"  - 召回率: {test_metrics['recall']:.4f}")
    print(f"  - F1分数: {test_metrics['f1']:.4f}")
    print(f"  - AUC-ROC: {test_metrics['auc_roc']:.4f}")
    print(f"\n训练时间统计:")
    print(f"  - 总训练时间（仅训练集）: {total_training_time:.3f}秒")
    print(f"  - 总验证时间: {total_val_time:.3f}秒")
    print(f"  - 训练+验证总时间: {total_time_with_val:.3f}秒")
    print(f"  - 平均训练时间（每epoch，仅训练集）: {avg_training_time:.3f}秒")
    print(f"  - 每个图的平均执行时间（整个训练过程）: {time_per_graph:.6f}秒/图 ({time_per_graph*1000:.3f}毫秒/图)")
    print(f"  - 每个图的平均执行时间（单个epoch）: {time_per_graph_per_epoch:.6f}秒/图 ({time_per_graph_per_epoch*1000:.3f}毫秒/图)")
    print(f"  - 训练集图数: {num_train_graphs}")
    print(f"  - 测试集推理时间: {test_inference_time:.3f}秒")
    print(f"  - 测试集每个图推理时间: {test_inference_time/num_test_graphs*1000:.3f}毫秒/图" if num_test_graphs > 0 else "  - 测试集每个图推理时间: N/A")
    print(f"  - 脚本总运行时间: {total_elapsed_time:.2f}秒")
    
    # 打印资源使用统计
    if resource_log['cpu_memory']:
        print(f"\n资源使用统计:")
        print(f"  - 峰值内存使用: {np.max(resource_log['cpu_memory']):.2f} MB")
        print(f"  - 平均内存使用: {np.mean(resource_log['cpu_memory']):.2f} MB")
        if resource_log['gpu_memory_max_allocated']:
            print(f"  - 峰值显存使用: {np.max(resource_log['gpu_memory_max_allocated']):.2f} MB")
            print(f"  - 平均显存使用: {np.mean(resource_log['gpu_memory_allocated']):.2f} MB")
    
    print(f"{'=' * 80}")
    
    # 保存详细结果
    if args.save_results:
        result_dir = setup_result_directory(args.model, args.data_source, args.dataset)
        
        # 保存训练时间统计到TXT文件
        time_stats_file = f'{result_dir}/training_time_stats.txt'
        with open(time_stats_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("训练时间统计报告\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"模型: {args.model}\n")
            f.write(f"数据集: {args.dataset}\n")
            f.write(f"数据源: {args.data_source}\n")
            f.write(f"随机种子: {args.seed}\n\n")
            f.write("-" * 80 + "\n")
            f.write("核心训练时间指标\n")
            f.write("-" * 80 + "\n")
            f.write(f"1. 模型到达收敛的总训练时间: {total_training_time:.3f} 秒\n")
            f.write(f"2. 平均训练时间（每epoch）: {avg_training_time:.3f} 秒\n\n")
            f.write("-" * 80 + "\n")
            f.write("详细训练信息\n")
            f.write("-" * 80 + "\n")
            f.write(f"实际训练轮数: {actual_epochs} epochs\n")
            f.write(f"最佳验证轮数: {best_epoch} epoch\n")
            f.write(f"最快epoch时间: {min(history['epoch_times']):.3f} 秒\n" if history['epoch_times'] else "最快epoch时间: 0.000 秒\n")
            f.write(f"最慢epoch时间: {max(history['epoch_times']):.3f} 秒\n" if history['epoch_times'] else "最慢epoch时间: 0.000 秒\n")
            f.write(f"epoch时间标准差: {np.std(history['epoch_times']):.3f} 秒\n" if history['epoch_times'] else "epoch时间标准差: 0.000 秒\n")
            f.write(f"脚本总运行时间: {total_elapsed_time:.2f} 秒\n\n")
            f.write("-" * 80 + "\n")
            f.write("测试集性能指标\n")
            f.write("-" * 80 + "\n")
            f.write(f"准确率: {test_metrics['accuracy']:.4f}\n")
            f.write(f"精确率: {test_metrics['precision']:.4f}\n")
            f.write(f"召回率: {test_metrics['recall']:.4f}\n")
            f.write(f"F1分数: {test_metrics['f1']:.4f}\n")
            f.write(f"AUC-ROC: {test_metrics['auc_roc']:.4f}\n")
            f.write(f"最佳验证F1: {best_val_f1:.4f}\n\n")
            f.write("-" * 80 + "\n")
            f.write("推理时间统计\n")
            f.write("-" * 80 + "\n")
            f.write(f"测试集推理时间: {test_inference_time:.3f} 秒\n")
            f.write(f"测试集图数量: {num_test_graphs}\n")
            f.write(f"每个图推理时间: {test_inference_time/num_test_graphs*1000:.3f} 毫秒/图\n" if num_test_graphs > 0 else "每个图推理时间: N/A\n")
            f.write("\n")
            f.write("=" * 80 + "\n")
        
        print(f"\n训练时间统计已保存到: {time_stats_file}")
        
        # 绘制图表
        plot_training_metrics(history['train'], history['val'], result_dir=result_dir)
        plot_confusion_matrix(
            test_metrics['labels'], 
            test_metrics['predictions'], 
            class_names=class_names,
            result_dir=result_dir
        )
        # 绘制累积时间曲线
        # 绘制训练时间图（使用训练集时间）
        plot_epoch_time(history['train_times'], history['cumulative_train_times'], result_dir=result_dir)
        
        # 保存训练历史到CSV文件（方便后续直接画图）
        save_training_history_csv(history, result_dir=result_dir)
        
        # 保存资源使用日志（只使用训练集时间）
        save_resource_usage_log(resource_log, num_train_graphs, total_training_time, avg_training_time, result_dir=result_dir)
        
        # 保存结果JSON
        result = {
            "experiment_info": {
                "data_source": args.data_source,
                "dataset": args.dataset,
                "model": args.model,
                "seed": args.seed,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "total_params": total_params,
                "best_epoch": best_epoch,
                "actual_epochs": actual_epochs,
                "total_elapsed_time": total_elapsed_time
            },
            "test_metrics": {
                "accuracy": float(test_metrics['accuracy']),
                "precision": float(test_metrics['precision']),
                "recall": float(test_metrics['recall']),
                "f1": float(test_metrics['f1']),
                "auc_roc": float(test_metrics['auc_roc'])
            },
            "inference_time": {
                "test_inference_time_seconds": float(test_inference_time),
                "test_inference_time_per_graph_ms": float(test_inference_time / num_test_graphs * 1000) if num_test_graphs > 0 else 0.0,
                "num_test_graphs": num_test_graphs
            },
            "best_val_f1": float(best_val_f1),
            "history": {
                "train_loss": history['train']['loss'],
                "train_f1": history['train']['f1'],
                "val_f1": history['val']['f1'],
                "train_times": history['train_times'],  # 每epoch训练时间（仅训练集）
                "val_times": history['val_times'],  # 每epoch验证时间
                "epoch_times": history['epoch_times'],  # 每epoch总时间（训练+验证）
                "cumulative_train_times": history['cumulative_train_times']  # 累积训练时间（仅训练集）
            },
            "time_statistics": {
                "total_training_time": float(total_training_time),  # 核心指标1: 总训练时间（仅训练集）
                "total_val_time": float(total_val_time),  # 总验证时间
                "total_time_with_val": float(total_time_with_val),  # 训练+验证总时间
                "avg_training_time_per_epoch": float(avg_training_time),  # 核心指标2: 平均每epoch训练时间（仅训练集）
                "time_per_graph": float(time_per_graph),  # 核心指标3: 每个图的平均执行时间（整个训练过程）
                "time_per_graph_ms": float(time_per_graph * 1000),  # 每个图的平均执行时间（毫秒，整个训练过程）
                "time_per_graph_per_epoch": float(time_per_graph_per_epoch),  # 核心指标4: 每个图在单个epoch中的平均执行时间
                "time_per_graph_per_epoch_ms": float(time_per_graph_per_epoch * 1000),  # 每个图在单个epoch中的平均执行时间（毫秒）
                "num_train_graphs": num_train_graphs,
                "num_total_graphs": total_graphs,
                "actual_epochs": actual_epochs,
                "avg_train_time_per_epoch": float(np.mean(history['train_times'])) if history['train_times'] else 0,
                "avg_val_time_per_epoch": float(np.mean(history['val_times'])) if history['val_times'] else 0,
                "avg_epoch_time": float(np.mean(history['epoch_times'])) if history['epoch_times'] else 0,
                "min_epoch_time": float(np.min(history['epoch_times'])) if history['epoch_times'] else 0,
                "max_epoch_time": float(np.max(history['epoch_times'])) if history['epoch_times'] else 0,
                "std_epoch_time": float(np.std(history['epoch_times'])) if history['epoch_times'] else 0,
                "total_cumulative_time": float(history['cumulative_train_times'][-1]) if history['cumulative_train_times'] else 0,
                "total_elapsed_time": float(total_elapsed_time)
            },
            "resource_usage": {
                "cpu_memory": {
                    "average_mb": float(np.mean(resource_log['cpu_memory'])) if resource_log['cpu_memory'] else 0,
                    "peak_mb": float(np.max(resource_log['cpu_memory'])) if resource_log['cpu_memory'] else 0,
                    "min_mb": float(np.min(resource_log['cpu_memory'])) if resource_log['cpu_memory'] else 0
                },
                "gpu_memory": {
                    "average_allocated_mb": float(np.mean(resource_log['gpu_memory_allocated'])) if resource_log['gpu_memory_allocated'] else 0,
                    "peak_allocated_mb": float(np.max(resource_log['gpu_memory_max_allocated'])) if resource_log['gpu_memory_max_allocated'] else 0,
                    "average_reserved_mb": float(np.mean(resource_log['gpu_memory_reserved'])) if resource_log['gpu_memory_reserved'] else 0,
                    "peak_reserved_mb": float(np.max(resource_log['gpu_memory_reserved'])) if resource_log['gpu_memory_reserved'] else 0
                }
            },
            "classification_report": classification_report(
                test_metrics['labels'],
                test_metrics['predictions'],
                target_names=class_names,
                output_dict=True,
                zero_division=0
            )
        }
        
        with open(f'{result_dir}/experiment_results.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        
        print(f"\n结果已保存到: {result_dir}")
    
    # 返回测试结果和时间统计
    return {
        **test_metrics,
        'actual_epochs': actual_epochs,
        'total_training_time': float(total_training_time),  # 核心指标1: 总训练时间
        'avg_training_time': float(avg_training_time),  # 核心指标2: 平均训练时间
        'time_per_graph': float(time_per_graph),  # 核心指标3: 每个图的平均执行时间（整个训练过程）
        'time_per_graph_ms': float(time_per_graph * 1000),
        'time_per_graph_per_epoch': float(time_per_graph_per_epoch),  # 核心指标4: 每个图在单个epoch中的平均执行时间
        'time_per_graph_per_epoch_ms': float(time_per_graph_per_epoch * 1000),
        'test_inference_time': float(test_inference_time),  # 测试集推理时间
        'test_inference_time_per_graph_ms': float(test_inference_time / num_test_graphs * 1000) if num_test_graphs > 0 else 0.0,
        'num_train_graphs': num_train_graphs,
        'num_test_graphs': num_test_graphs,
        'avg_epoch_time': float(np.mean(history['epoch_times'])) if history['epoch_times'] else 0,
        'total_elapsed_time': total_elapsed_time,
        'peak_cpu_memory_mb': float(np.max(resource_log['cpu_memory'])) if resource_log['cpu_memory'] else 0,
        'peak_gpu_memory_mb': float(np.max(resource_log['gpu_memory_max_allocated'])) if resource_log['gpu_memory_max_allocated'] else 0,
        'avg_cpu_memory_mb': float(np.mean(resource_log['cpu_memory'])) if resource_log['cpu_memory'] else 0,
        'avg_gpu_memory_mb': float(np.mean(resource_log['gpu_memory_allocated'])) if resource_log['gpu_memory_allocated'] else 0
    }


def main():
    # 获取脚本所在目录，用于构建正确的相对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_root = os.path.join(script_dir, '../../data/TUDataset')
    default_ogb_root = os.path.join(script_dir, '../../data/OGB')
    
    parser = argparse.ArgumentParser(
        description='通用图分类模型训练 - 支持CSV数据、TUDataset和OGB数据集',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # TUDataset（标准基准测试）
  python run.py --data_source tudataset --dataset MUTAG --model gcn
  
  # OGB数据集（大规模图基准测试）
  python run.py --data_source ogb --dataset ogbg-molhiv --model gin --gpu_id 1
  
  # CSV数据（神经元数据）
  python run.py --data_source csv --dataset dataset/processed3.csv --model gcn
  
  # 自定义图数据集（如random数据集）- 使用随机特征
  python run.py --data_source custom --dataset /path/to/graphs_100 --model gcn --feature_dim 16
  
  # 自定义图数据集 - 使用节点度数特征
  python run.py --data_source custom --dataset /path/to/graphs_100 --model gcn --use_degree_feature
  
  # Random数据集完整示例（4个规模）
  python run.py --data_source custom --dataset ../../data/random/graphs_100 --model gin --save_results
  python run.py --data_source custom --dataset ../../data/random/graphs_1000 --model gin --save_results
  python run.py --data_source custom --dataset ../../data/random/graphs_10000 --model gin --save_results
  python run.py --data_source custom --dataset ../../data/random/graphs_100000 --model gin --save_results
  
  # 20次重复训练（每次seed递增，数据划分不同）
  python run.py --data_source tudataset --dataset MUTAG --model gcn --num_runs 20 --seed 42
  
  # 自定义数据集划分比例（训练60%，验证20%，测试20%）
  python run.py --data_source tudataset --dataset MUTAG --model gcn --train_ratio 0.6 --val_ratio 0.2
  
  # 完整示例：20次训练 + 自定义比例 + 保存结果
  python run.py --data_source tudataset --dataset MUTAG --model gcn \
                --num_runs 20 --seed 42 \
                --train_ratio 0.6 --val_ratio 0.2 \
                --save_results --gpu_id 1
        """
    )
    
    # 数据源参数
    parser.add_argument('--data_source', type=str, required=True,
                        choices=['csv', 'tudataset', 'ogb', 'custom'],
                        help='数据源类型: csv (CSV文件), tudataset (TUDataset), ogb (OGB数据集), custom (自定义图数据集)')
    parser.add_argument('--dataset', type=str, required=True,
                        help='数据集名称或路径')
    parser.add_argument('--data_root', type=str, default=default_data_root,
                        help='TUDataset根目录（仅用于tudataset）')
    parser.add_argument('--ogb_root', type=str, default=default_ogb_root,
                        help='OGB数据集根目录（仅用于ogb）')
    parser.add_argument('--train_ratio', type=float, default=0.6,
                        help='训练集比例 (默认60%)')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='验证集比例 (默认20%, 测试集也是20%)')
    
    # 自定义数据集参数（仅用于custom数据源）
    parser.add_argument('--feature_dim', type=int, default=16,
                        help='随机节点特征维度（仅用于custom数据源，默认16）')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='类别数（仅用于custom数据源，默认2）')
    parser.add_argument('--use_degree_feature', action='store_true',
                        help='使用节点度数作为特征（仅用于custom数据源）')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='gcn',
                        choices=['gcn', 'gat', 'sage', 'hybrid', 'gin', 'chebnet', 'edgeconv', 'gunet', 
                                 'pna', 'gatv2', 'deepergcn'],
                        help='模型类型')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='隐藏层维度')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout比例')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批大小')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='权重衰减')
    parser.add_argument('--epochs', type=int, default=10,
                        help='最大训练轮数')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping耐心值')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（初始值）')
    parser.add_argument('--num_runs', type=int, default=1,
                        help='重复训练次数（每次seed递增，确保数据划分不同）')
    parser.add_argument('--print_every', type=int, default=10,
                        help='打印间隔')
    parser.add_argument('--gpu_id', type=int, default=3,
                        help='使用的GPU ID (例如：0, 1, 2)，-1表示使用默认GPU')
    
    # 保存参数
    parser.add_argument('--save_results', action='store_true',
                        help='保存详细结果和图表')
    parser.add_argument('--save_model', action='store_true',
                        help='保存最佳模型')
    
    args = parser.parse_args()
    
    # 多次运行实验（每次seed递增）
    if args.num_runs > 1:
        print("\n" + "=" * 80)
        print(f"将进行 {args.num_runs} 次独立训练实验")
        print(f"初始seed: {args.seed}, 每次递增1")
        print(f"数据集划分比例: 训练集{args.train_ratio*100:.0f}%, 验证集{args.val_ratio*100:.0f}%, 测试集{(1-args.train_ratio-args.val_ratio)*100:.0f}%")
        print("=" * 80 + "\n")
        
        all_results = []
        initial_seed = args.seed  # 保存初始seed
        
        for run_idx in range(args.num_runs):
            current_seed = initial_seed + run_idx  # 使用初始seed
            args.seed = current_seed
            
            print("\n" + "🔄" * 40)
            print(f"🚀 开始第 {run_idx + 1}/{args.num_runs} 次训练 (Seed = {current_seed})")
            print("🔄" * 40 + "\n")
            
            result = run_experiment(args)
            all_results.append({
                'run': run_idx + 1,
                'seed': current_seed,
                'test_accuracy': float(result['accuracy']),
                'test_precision': float(result['precision']),
                'test_recall': float(result['recall']),
                'test_f1': float(result['f1']),
                'test_auc_roc': float(result['auc_roc']),
                'actual_epochs': int(result['actual_epochs']),
                'total_training_time': float(result['total_training_time']),
                'avg_training_time': float(result['avg_training_time']),
                'time_per_graph': float(result['time_per_graph']),
                'time_per_graph_ms': float(result['time_per_graph_ms']),
                'time_per_graph_per_epoch': float(result['time_per_graph_per_epoch']),
                'time_per_graph_per_epoch_ms': float(result['time_per_graph_per_epoch_ms']),
                'test_inference_time': float(result['test_inference_time']),
                'test_inference_time_per_graph_ms': float(result['test_inference_time_per_graph_ms']),
                'num_train_graphs': int(result['num_train_graphs']),
                'num_test_graphs': int(result['num_test_graphs']),
                'avg_epoch_time': float(result['avg_epoch_time']),
                'total_elapsed_time': float(result['total_elapsed_time']),
                'peak_cpu_memory_mb': float(result['peak_cpu_memory_mb']),
                'peak_gpu_memory_mb': float(result['peak_gpu_memory_mb']),
                'avg_cpu_memory_mb': float(result['avg_cpu_memory_mb']),
                'avg_gpu_memory_mb': float(result['avg_gpu_memory_mb'])
            })
        
        # 打印汇总结果
        print("\n" + "=" * 80)
        print("📊 多次训练汇总结果")
        print("=" * 80)
        
        accuracies = [r['test_accuracy'] for r in all_results]
        precisions = [r['test_precision'] for r in all_results]
        recalls = [r['test_recall'] for r in all_results]
        f1_scores = [r['test_f1'] for r in all_results]
        auc_rocs = [r['test_auc_roc'] for r in all_results]
        actual_epochs = [r['actual_epochs'] for r in all_results]
        total_training_times = [r['total_training_time'] for r in all_results]
        avg_training_times = [r['avg_training_time'] for r in all_results]
        times_per_graph = [r['time_per_graph'] for r in all_results]
        times_per_graph_ms = [r['time_per_graph_ms'] for r in all_results]
        times_per_graph_per_epoch = [r['time_per_graph_per_epoch'] for r in all_results]
        times_per_graph_per_epoch_ms = [r['time_per_graph_per_epoch_ms'] for r in all_results]
        test_inference_times = [r['test_inference_time'] for r in all_results]
        test_inference_times_per_graph_ms = [r['test_inference_time_per_graph_ms'] for r in all_results]
        avg_epoch_times = [r['avg_epoch_time'] for r in all_results]
        total_elapsed_times = [r['total_elapsed_time'] for r in all_results]
        peak_cpu_mems = [r['peak_cpu_memory_mb'] for r in all_results]
        peak_gpu_mems = [r['peak_gpu_memory_mb'] for r in all_results]
        avg_cpu_mems = [r['avg_cpu_memory_mb'] for r in all_results]
        avg_gpu_mems = [r['avg_gpu_memory_mb'] for r in all_results]
        
        print(f"\n测试集准确率: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"测试集精确率: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
        print(f"测试集召回率: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
        print(f"测试集F1分数: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
        print(f"测试集AUC-ROC: {np.mean(auc_rocs):.4f} ± {np.std(auc_rocs):.4f}")
        print(f"\n训练时间统计:")
        print(f"平均实际训练轮数: {np.mean(actual_epochs):.1f} ± {np.std(actual_epochs):.1f} epochs")
        print(f"平均总训练时间（到达收敛）: {np.mean(total_training_times):.3f} ± {np.std(total_training_times):.3f}秒")
        print(f"平均训练时间（每epoch）: {np.mean(avg_training_times):.3f} ± {np.std(avg_training_times):.3f}秒")
        print(f"平均每个图执行时间: {np.mean(times_per_graph):.6f} ± {np.std(times_per_graph):.6f}秒/图")
        print(f"平均每个图执行时间: {np.mean(times_per_graph_ms):.3f} ± {np.std(times_per_graph_ms):.3f}毫秒/图")
        print(f"平均epoch时间: {np.mean(avg_epoch_times):.3f} ± {np.std(avg_epoch_times):.3f}秒")
        print(f"平均脚本运行时间: {np.mean(total_elapsed_times):.2f} ± {np.std(total_elapsed_times):.2f}秒")
        
        print(f"\n推理时间统计:")
        print(f"平均测试集推理时间: {np.mean(test_inference_times):.3f} ± {np.std(test_inference_times):.3f}秒")
        print(f"平均每个图推理时间: {np.mean(test_inference_times_per_graph_ms):.3f} ± {np.std(test_inference_times_per_graph_ms):.3f}毫秒/图")
        
        print(f"\n资源使用统计:")
        print(f"平均峰值内存: {np.mean(peak_cpu_mems):.2f} ± {np.std(peak_cpu_mems):.2f} MB")
        print(f"平均峰值显存: {np.mean(peak_gpu_mems):.2f} ± {np.std(peak_gpu_mems):.2f} MB")
        print(f"平均内存使用: {np.mean(avg_cpu_mems):.2f} ± {np.std(avg_cpu_mems):.2f} MB")
        print(f"平均显存使用: {np.mean(avg_gpu_mems):.2f} ± {np.std(avg_gpu_mems):.2f} MB")
        
        print(f"\n详细结果:")
        for result in all_results:
            print(f"  Run {result['run']} (Seed={result['seed']}): "
                  f"Acc={result['test_accuracy']:.4f}, "
                  f"F1={result['test_f1']:.4f}, "
                  f"AUC-ROC={result['test_auc_roc']:.4f}, "
                  f"Epochs={result['actual_epochs']}, "
                  f"TrainTime={result['total_training_time']:.3f}s, "
                  f"InferTime={result['test_inference_time']:.3f}s, "
                  f"TimePerGraph={result['time_per_graph_ms']:.3f}ms, "
                  f"PeakMem={result['peak_cpu_memory_mb']:.1f}MB, "
                  f"PeakGPU={result['peak_gpu_memory_mb']:.1f}MB")
        
        # 保存汇总结果
        summary_file = f"result/multi_run_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs('result', exist_ok=True)
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                'num_runs': args.num_runs,
                'initial_seed': initial_seed,
                'statistics': {
                    'accuracy': {'mean': float(np.mean(accuracies)), 'std': float(np.std(accuracies))},
                    'precision': {'mean': float(np.mean(precisions)), 'std': float(np.std(precisions))},
                    'recall': {'mean': float(np.mean(recalls)), 'std': float(np.std(recalls))},
                    'f1': {'mean': float(np.mean(f1_scores)), 'std': float(np.std(f1_scores))},
                    'auc_roc': {'mean': float(np.mean(auc_rocs)), 'std': float(np.std(auc_rocs))},
                    'actual_epochs': {'mean': float(np.mean(actual_epochs)), 'std': float(np.std(actual_epochs))},
                    'total_training_time': {'mean': float(np.mean(total_training_times)), 'std': float(np.std(total_training_times))},
                    'avg_training_time_per_epoch': {'mean': float(np.mean(avg_training_times)), 'std': float(np.std(avg_training_times))},
                    'time_per_graph': {'mean': float(np.mean(times_per_graph)), 'std': float(np.std(times_per_graph))},
                    'time_per_graph_ms': {'mean': float(np.mean(times_per_graph_ms)), 'std': float(np.std(times_per_graph_ms))},
                    'time_per_graph_per_epoch': {'mean': float(np.mean(times_per_graph_per_epoch)), 'std': float(np.std(times_per_graph_per_epoch))},
                    'time_per_graph_per_epoch_ms': {'mean': float(np.mean(times_per_graph_per_epoch_ms)), 'std': float(np.std(times_per_graph_per_epoch_ms))},
                    'test_inference_time': {'mean': float(np.mean(test_inference_times)), 'std': float(np.std(test_inference_times))},
                    'test_inference_time_per_graph_ms': {'mean': float(np.mean(test_inference_times_per_graph_ms)), 'std': float(np.std(test_inference_times_per_graph_ms))},
                    'avg_epoch_time': {'mean': float(np.mean(avg_epoch_times)), 'std': float(np.std(avg_epoch_times))},
                    'total_elapsed_time': {'mean': float(np.mean(total_elapsed_times)), 'std': float(np.std(total_elapsed_times))},
                    'peak_cpu_memory_mb': {'mean': float(np.mean(peak_cpu_mems)), 'std': float(np.std(peak_cpu_mems))},
                    'peak_gpu_memory_mb': {'mean': float(np.mean(peak_gpu_mems)), 'std': float(np.std(peak_gpu_mems))},
                    'avg_cpu_memory_mb': {'mean': float(np.mean(avg_cpu_mems)), 'std': float(np.std(avg_cpu_mems))},
                    'avg_gpu_memory_mb': {'mean': float(np.mean(avg_gpu_mems)), 'std': float(np.std(avg_gpu_mems))}
                },
                'all_results': all_results
            }, f, ensure_ascii=False, indent=4)
        
        print(f"\n汇总结果已保存到: {summary_file}")
        print("=" * 80 + "\n")
    else:
        # 单次运行
        run_experiment(args)


if __name__ == "__main__":
    main()

