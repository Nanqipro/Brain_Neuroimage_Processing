#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
通用图分类模型训练脚本

支持两种数据源：
1. 原始CSV数据（神经元数据）- 需要构建图
2. TUDataset标准数据集 - 已有图结构

使用方法：
    # 使用TUDataset（推荐用于标准基准测试）
    python run.py --data_source tudataset --dataset MUTAG --model gcn
    
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


# ============================================================================
# 训练函数
# ============================================================================

def setup_result_directory(model_name, data_source, dataset_name):
    """创建结果保存目录"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if data_source == 'csv':
        base_dir = f"result/csv/{model_name}"
    else:
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
    else:  # tudataset
        train_loader, val_loader, test_loader, num_features, num_classes, _, _ = \
            load_tudataset_data(
                args.dataset, args.data_root, args.seed,
                args.train_ratio, args.val_ratio, args.batch_size
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
        optimizer, mode='max', factor=0.5, patience=10, verbose=False
    )
    
    # 训练历史
    history = {
        'train': {'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
        'val': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    }
    
    best_val_f1 = 0
    best_epoch = 0
    patience_counter = 0
    
    print(f"\n开始训练 (最多 {args.epochs} 轮)...")
    print("=" * 80)
    
    # 训练循环
    for epoch in range(1, args.epochs + 1):
        # 训练
        train_metrics = train_model(model, train_loader, optimizer, device, class_weights)
        
        # 验证
        val_metrics = evaluate_model(model, val_loader, device)
        
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
                  f"Val Acc={val_metrics['accuracy']:.4f}")
        
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
                "val_f1": history['val']['f1']
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
    
    return test_metrics


def main():
    parser = argparse.ArgumentParser(
        description='通用图分类模型训练 - 支持CSV数据和TUDataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # TUDataset（标准基准测试）
  python run.py --data_source tudataset --dataset MUTAG --model gcn
  
  # CSV数据（神经元数据）
  python run.py --data_source csv --dataset dataset/processed3.csv --model gcn
  
  # 保存完整结果
  python run.py --data_source tudataset --dataset MUTAG --model gcn --save_results
        """
    )
    
    # 数据源参数
    parser.add_argument('--data_source', type=str, required=True,
                        choices=['csv', 'tudataset'],
                        help='数据源类型: csv (CSV文件), tudataset (TUDataset)')
    parser.add_argument('--dataset', type=str, required=True,
                        help='数据集名称或CSV文件路径')
    parser.add_argument('--data_root', type=str, default='../../data/TUDataset',
                        help='TUDataset根目录（仅用于tudataset）')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='验证集比例')
    
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
    parser.add_argument('--epochs', type=int, default=200,
                        help='最大训练轮数')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping耐心值')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--print_every', type=int, default=10,
                        help='打印间隔')
    
    # 保存参数
    parser.add_argument('--save_results', action='store_true',
                        help='保存详细结果和图表')
    parser.add_argument('--save_model', action='store_true',
                        help='保存最佳模型')
    
    args = parser.parse_args()
    
    run_experiment(args)


if __name__ == "__main__":
    main()

