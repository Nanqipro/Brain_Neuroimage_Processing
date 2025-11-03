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
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

# 导入模型和工具函数
from model import (
    ImprovedGCN, PureGCN, PureGAT, PureGraphSAGE,
    GIN, ChebNet, EdgeConvNet, GraphUNet,
    PNA, GATv2, DeeperGCN
)
from train import train_model, evaluate_model, plot_confusion_matrix, plot_training_metrics
from process import (load_data, oversample_data, compute_correlation_matrix, 
                     create_pyg_dataset, visualize_graph)

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
        'epoch_time': history['epoch_times'],
        'cumulative_time': history['cumulative_times'],
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
    
    # 创建子数据集并转换特征类型
    # OGB数据集的节点特征通常是int64类型，需要转换为float32以兼容所有GNN模型
    def convert_data_to_float(data):
        """将图数据的节点特征转换为float32类型"""
        if data.x is not None and data.x.dtype != torch.float32:
            data.x = data.x.float()
        return data
    
    train_dataset = [convert_data_to_float(dataset[i]) for i in split_idx['train']]
    val_dataset = [convert_data_to_float(dataset[i]) for i in split_idx['valid']]
    test_dataset = [convert_data_to_float(dataset[i]) for i in split_idx['test']]
    
    print(f"\n数据集划分 (预定义):")
    print(f"  - 训练集: {len(train_dataset)} 图")
    print(f"  - 验证集: {len(val_dataset)} 图")
    print(f"  - 测试集: {len(test_dataset)} 图")
    print(f"  - 节点特征已转换为float32类型")
    
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
    
    return train_loader, val_loader, test_loader, dataset.num_features, num_classes, None, None


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
    else:  # ogb
        train_loader, val_loader, test_loader, num_features, num_classes, _, _ = \
            load_ogb_data(
                args.dataset, args.ogb_root, args.batch_size
            )
        class_weights = None
        class_names = [str(i) for i in range(num_classes)]
    
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
    
    # 优化器和学习率调度器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=20
    )
    
    # 训练历史
    history = {
        'train': {'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
        'val': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
        'epoch_times': [],  # 记录每个epoch的单独执行时间
        'cumulative_times': []  # 记录从训练开始的累积时间
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
        # 记录epoch开始时间（只记录模型执行时间）
        epoch_start_time = time.time()
        
        # 训练
        train_metrics = train_model(model, train_loader, optimizer, device, class_weights)
        
        # 验证
        val_metrics = evaluate_model(model, val_loader, device)
        
        # 记录epoch结束时间（包含训练和验证的模型执行时间）
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        history['epoch_times'].append(epoch_time)
        
        # 记录累积时间（从训练开始到当前epoch结束）
        cumulative_time = epoch_end_time - training_start_time
        history['cumulative_times'].append(cumulative_time)
        
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
                  f"Val Acc={val_metrics['accuracy']:.4f}, Time={epoch_time:.3f}s")
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    print("=" * 80)
    
    # 测试
    print(f"\n在测试集上评估...")
    test_metrics = evaluate_model(model, test_loader, device)
    
    # 计算训练时间
    training_time = time.time() - begin_time
    
    # 打印结果
    print(f"\n{'=' * 80}")
    print(f"实验结果:")
    print(f"{'=' * 80}")
    print(f"最佳验证F1 (Epoch {best_epoch}): {best_val_f1:.4f}")
    print(f"\n测试集性能:")
    print(f"  - 准确率: {test_metrics['accuracy']:.4f}")
    print(f"  - 精确率: {test_metrics['precision']:.4f}")
    print(f"  - 召回率: {test_metrics['recall']:.4f}")
    print(f"  - F1分数: {test_metrics['f1']:.4f}")
    print(f"\n训练时间: {training_time:.2f}秒")
    print(f"{'=' * 80}")
    
    # 保存详细结果
    if args.save_results:
        result_dir = setup_result_directory(args.model, args.data_source, args.dataset)
        
        # 绘制图表
        plot_training_metrics(history['train'], history['val'], result_dir=result_dir)
        plot_confusion_matrix(
            test_metrics['labels'], 
            test_metrics['predictions'], 
            class_names=class_names,
            result_dir=result_dir
        )
        # 绘制累积时间曲线
        plot_epoch_time(history['epoch_times'], history['cumulative_times'], result_dir=result_dir)
        
        # 保存训练历史到CSV文件（方便后续直接画图）
        save_training_history_csv(history, result_dir=result_dir)
        
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
                "training_time": training_time
            },
            "test_metrics": {
                "accuracy": float(test_metrics['accuracy']),
                "precision": float(test_metrics['precision']),
                "recall": float(test_metrics['recall']),
                "f1": float(test_metrics['f1'])
            },
            "best_val_f1": float(best_val_f1),
            "history": {
                "train_loss": history['train']['loss'],
                "train_f1": history['train']['f1'],
                "val_f1": history['val']['f1'],
                "epoch_times": history['epoch_times'],
                "cumulative_times": history['cumulative_times']
            },
            "time_statistics": {
                "avg_epoch_time": float(np.mean(history['epoch_times'])) if history['epoch_times'] else 0,
                "min_epoch_time": float(np.min(history['epoch_times'])) if history['epoch_times'] else 0,
                "max_epoch_time": float(np.max(history['epoch_times'])) if history['epoch_times'] else 0,
                "std_epoch_time": float(np.std(history['epoch_times'])) if history['epoch_times'] else 0,
                "total_cumulative_time": float(history['cumulative_times'][-1]) if history['cumulative_times'] else 0
            },
            "classification_report": classification_report(
                test_metrics['labels'],
                test_metrics['predictions'],
                target_names=class_names,
                output_dict=True
            )
        }
        
        with open(f'{result_dir}/experiment_results.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        
        print(f"\n结果已保存到: {result_dir}")
    
    # 返回测试结果和时间统计
    return {
        **test_metrics,
        'avg_epoch_time': float(np.mean(history['epoch_times'])) if history['epoch_times'] else 0,
        'total_training_time': training_time
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
                        choices=['csv', 'tudataset', 'ogb'],
                        help='数据源类型: csv (CSV文件), tudataset (TUDataset), ogb (OGB数据集)')
    parser.add_argument('--dataset', type=str, required=True,
                        help='数据集名称或CSV文件路径')
    parser.add_argument('--data_root', type=str, default=default_data_root,
                        help='TUDataset根目录（仅用于tudataset）')
    parser.add_argument('--ogb_root', type=str, default=default_ogb_root,
                        help='OGB数据集根目录（仅用于ogb）')
    parser.add_argument('--train_ratio', type=float, default=0.6,
                        help='训练集比例 (默认60%)')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='验证集比例 (默认20%, 测试集也是20%)')
    
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
                'avg_epoch_time': float(result['avg_epoch_time']),
                'total_training_time': float(result['total_training_time'])
            })
        
        # 打印汇总结果
        print("\n" + "=" * 80)
        print("📊 多次训练汇总结果")
        print("=" * 80)
        
        accuracies = [r['test_accuracy'] for r in all_results]
        precisions = [r['test_precision'] for r in all_results]
        recalls = [r['test_recall'] for r in all_results]
        f1_scores = [r['test_f1'] for r in all_results]
        avg_epoch_times = [r['avg_epoch_time'] for r in all_results]
        total_times = [r['total_training_time'] for r in all_results]
        
        print(f"\n测试集准确率: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"测试集精确率: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
        print(f"测试集召回率: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
        print(f"测试集F1分数: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
        print(f"\n时间统计:")
        print(f"平均每轮epoch时间: {np.mean(avg_epoch_times):.3f} ± {np.std(avg_epoch_times):.3f}秒")
        print(f"平均总训练时间: {np.mean(total_times):.2f} ± {np.std(total_times):.2f}秒")
        
        print(f"\n详细结果:")
        for result in all_results:
            print(f"  Run {result['run']} (Seed={result['seed']}): "
                  f"Acc={result['test_accuracy']:.4f}, "
                  f"Prec={result['test_precision']:.4f}, "
                  f"Recall={result['test_recall']:.4f}, "
                  f"F1={result['test_f1']:.4f}, "
                  f"AvgEpochTime={result['avg_epoch_time']:.3f}s, "
                  f"TotalTime={result['total_training_time']:.2f}s")
        
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
                    'avg_epoch_time': {'mean': float(np.mean(avg_epoch_times)), 'std': float(np.std(avg_epoch_times))},
                    'total_training_time': {'mean': float(np.mean(total_times)), 'std': float(np.std(total_times))}
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

