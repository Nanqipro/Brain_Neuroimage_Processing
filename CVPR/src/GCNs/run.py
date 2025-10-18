#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
通用图分类模型训练脚本

支持两种数据源：
1. 原始CSV数据（神经元数据）- 需要构建图
2. TUDataset标准数据集 - 已有图结构

新功能：
- 支持多数据集批量训练
- 支持配置文件批量运行
- 生成汇总报告

使用方法：
    # 单个数据集
    python run.py --data_source tudataset --dataset MUTAG --model gcn
    
    # 多个数据集（用逗号分隔）
    python run.py --data_source tudataset --dataset MUTAG,PROTEINS,DD --model gcn
    
    # 使用配置文件批量运行
    python run.py --config experiments_config.json
    
    # 使用CSV数据
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
from typing import List, Dict, Any

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


def load_common_data(dataset_name, data_root, seed, train_ratio=0.8, val_ratio=0.1, batch_size=32):
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
            load_common_data(
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


def load_config(config_path: str) -> Dict[str, Any]:
    """从JSON配置文件加载实验配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def run_multiple_datasets(args):
    """在多个数据集上运行实验并生成汇总报告"""
    
    # 解析数据集列表
    if ',' in args.dataset:
        datasets = [d.strip() for d in args.dataset.split(',')]
    else:
        datasets = [args.dataset]
    
    print("\n" + "=" * 80)
    print(f"将在 {len(datasets)} 个数据集上运行实验")
    print(f"数据集: {', '.join(datasets)}")
    print("=" * 80 + "\n")
    
    # 存储所有结果
    all_results = []
    
    # 对每个数据集运行实验
    for idx, dataset in enumerate(datasets, 1):
        print(f"\n{'#' * 80}")
        print(f"# 实验 {idx}/{len(datasets)}: {dataset}")
        print(f"{'#' * 80}\n")
        
        # 创建新的args副本
        dataset_args = argparse.Namespace(**vars(args))
        dataset_args.dataset = dataset
        
        try:
            # 运行实验
            test_metrics = run_experiment(dataset_args)
            
            # 记录结果
            result = {
                'dataset': dataset,
                'model': args.model,
                'accuracy': float(test_metrics['accuracy']),
                'precision': float(test_metrics['precision']),
                'recall': float(test_metrics['recall']),
                'f1': float(test_metrics['f1']),
                'status': 'success'
            }
            all_results.append(result)
            
        except Exception as e:
            print(f"\n❌ 数据集 {dataset} 训练失败: {str(e)}")
            result = {
                'dataset': dataset,
                'model': args.model,
                'status': 'failed',
                'error': str(e)
            }
            all_results.append(result)
        
        print(f"\n完成 {idx}/{len(datasets)}\n")
    
    # 生成汇总报告
    print("\n" + "=" * 80)
    print("实验汇总报告")
    print("=" * 80 + "\n")
    
    # 创建汇总表格
    summary_df = pd.DataFrame([r for r in all_results if r['status'] == 'success'])
    
    if len(summary_df) > 0:
        print(f"模型: {args.model}\n")
        print(summary_df[['dataset', 'accuracy', 'precision', 'recall', 'f1']].to_string(index=False))
        
        print(f"\n平均性能:")
        print(f"  - 准确率: {summary_df['accuracy'].mean():.4f} ± {summary_df['accuracy'].std():.4f}")
        print(f"  - 精确率: {summary_df['precision'].mean():.4f} ± {summary_df['precision'].std():.4f}")
        print(f"  - 召回率: {summary_df['recall'].mean():.4f} ± {summary_df['recall'].std():.4f}")
        print(f"  - F1分数: {summary_df['f1'].mean():.4f} ± {summary_df['f1'].std():.4f}")
    
    # 显示失败的实验
    failed = [r for r in all_results if r['status'] == 'failed']
    if failed:
        print(f"\n失败的实验 ({len(failed)}):")
        for r in failed:
            print(f"  - {r['dataset']}: {r['error']}")
    
    print("\n" + "=" * 80)
    
    # 保存汇总报告
    if args.save_results:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_dir = f"result/summary/{args.model}"
        os.makedirs(summary_dir, exist_ok=True)
        
        summary_path = f"{summary_dir}/summary_{timestamp}.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({
                'model': args.model,
                'datasets': datasets,
                'results': all_results,
                'timestamp': timestamp
            }, f, ensure_ascii=False, indent=4)
        
        # 保存CSV格式
        if len(summary_df) > 0:
            csv_path = f"{summary_dir}/summary_{timestamp}.csv"
            summary_df.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"\n汇总报告已保存到: {summary_dir}")
    
    return all_results


def run_from_config(config_path: str):
    """从配置文件运行批量实验"""
    
    config = load_config(config_path)
    print(f"\n从配置文件加载实验: {config_path}")
    print(f"实验数量: {len(config.get('experiments', []))}")
    
    all_results = []
    
    # 运行每个实验
    for idx, exp_config in enumerate(config.get('experiments', []), 1):
        print(f"\n{'#' * 80}")
        print(f"# 配置实验 {idx}/{len(config['experiments'])}")
        print(f"{'#' * 80}\n")
        
        # 创建args对象
        args = argparse.Namespace(**exp_config)
        
        # 设置默认值
        if not hasattr(args, 'save_results'):
            args.save_results = True
        if not hasattr(args, 'save_model'):
            args.save_model = False
        
        try:
            # 如果有多个数据集，使用批量运行
            if hasattr(args, 'dataset') and ',' in str(args.dataset):
                results = run_multiple_datasets(args)
                all_results.extend(results)
            else:
                test_metrics = run_experiment(args)
                result = {
                    'dataset': args.dataset,
                    'model': args.model,
                    'accuracy': float(test_metrics['accuracy']),
                    'precision': float(test_metrics['precision']),
                    'recall': float(test_metrics['recall']),
                    'f1': float(test_metrics['f1']),
                    'status': 'success'
                }
                all_results.append(result)
        except Exception as e:
            print(f"\n❌ 实验失败: {str(e)}")
            result = {
                'dataset': args.dataset if hasattr(args, 'dataset') else 'unknown',
                'model': args.model if hasattr(args, 'model') else 'unknown',
                'status': 'failed',
                'error': str(e)
            }
            all_results.append(result)
    
    # 保存总体汇总
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_dir = f"result/batch_summary"
    os.makedirs(summary_dir, exist_ok=True)
    
    summary_path = f"{summary_dir}/batch_results_{timestamp}.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'config_file': config_path,
            'total_experiments': len(all_results),
            'results': all_results,
            'timestamp': timestamp
        }, f, ensure_ascii=False, indent=4)
    
    print(f"\n批量实验结果已保存到: {summary_path}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='通用图分类模型训练 - 支持CSV数据、TUDataset和批量实验',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 单个TUDataset
  python run.py --data_source tudataset --dataset MUTAG --model gcn
  
  # 多个TUDataset（用逗号分隔）
  python run.py --data_source tudataset --dataset MUTAG,PROTEINS,DD --model gcn --save_results
  
  # CSV数据
  python run.py --data_source csv --dataset dataset/processed3.csv --model gcn
  
  # 使用配置文件批量运行
  python run.py --config experiments_config.json
  
  # 在所有常用数据集上测试模型
  python run.py --data_source tudataset --dataset MUTAG,ENZYMES,PROTEINS,COLLAB --model gin --save_results
        """
    )
    
    # 配置文件选项
    parser.add_argument('--config', type=str, default=None,
                        help='从JSON配置文件运行批量实验')
    
    # 数据源参数
    parser.add_argument('--data_source', type=str, default=None,
                        choices=['csv', 'tudataset'],
                        help='数据源类型: csv (CSV文件), tudataset (TUDataset)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='数据集名称或CSV文件路径（支持逗号分隔多个数据集）')
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
    
    # 如果指定了配置文件，从配置文件运行
    if args.config:
        run_from_config(args.config)
    # 如果dataset包含多个数据集，批量运行
    elif args.dataset and ',' in args.dataset:
        if not args.data_source:
            parser.error("使用多数据集时必须指定 --data_source")
        run_multiple_datasets(args)
    # 否则运行单个实验
    else:
        if not args.data_source or not args.dataset:
            parser.error("必须指定 --data_source 和 --dataset，或使用 --config")
        run_experiment(args)


if __name__ == "__main__":
    main()

