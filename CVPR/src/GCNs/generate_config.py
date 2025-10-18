#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验配置生成工具

快速生成批量实验的JSON配置文件

使用示例:
    # 生成模型对比配置
    python generate_config.py --type model_comparison --dataset MUTAG --output config.json
    
    # 生成数据集对比配置
    python generate_config.py --type dataset_comparison --model gcn --output config.json
    
    # 生成超参数搜索配置
    python generate_config.py --type hyperparam_search --dataset MUTAG --model gcn --output config.json
"""

import json
import argparse
from typing import List, Dict, Any


def generate_model_comparison_config(
    dataset: str,
    models: List[str],
    base_config: Dict[str, Any]
) -> Dict[str, Any]:
    """生成模型对比配置"""
    
    experiments = []
    for model in models:
        exp = base_config.copy()
        exp['dataset'] = dataset
        exp['model'] = model
        experiments.append(exp)
    
    return {
        "description": f"在 {dataset} 数据集上对比不同模型",
        "experiments": experiments
    }


def generate_dataset_comparison_config(
    datasets: List[str],
    model: str,
    base_config: Dict[str, Any]
) -> Dict[str, Any]:
    """生成数据集对比配置"""
    
    experiments = []
    for dataset in datasets:
        exp = base_config.copy()
        exp['dataset'] = dataset
        exp['model'] = model
        experiments.append(exp)
    
    return {
        "description": f"使用 {model} 模型在多个数据集上评估",
        "experiments": experiments
    }


def generate_hyperparam_search_config(
    dataset: str,
    model: str,
    hidden_dims: List[int],
    dropouts: List[float],
    lrs: List[float],
    base_config: Dict[str, Any]
) -> Dict[str, Any]:
    """生成超参数搜索配置（网格搜索）"""
    
    experiments = []
    for hidden_dim in hidden_dims:
        for dropout in dropouts:
            for lr in lrs:
                exp = base_config.copy()
                exp['dataset'] = dataset
                exp['model'] = model
                exp['hidden_dim'] = hidden_dim
                exp['dropout'] = dropout
                exp['lr'] = lr
                experiments.append(exp)
    
    return {
        "description": f"{model} 在 {dataset} 上的超参数搜索",
        "experiments": experiments
    }


def generate_seed_robustness_config(
    dataset: str,
    model: str,
    seeds: List[int],
    base_config: Dict[str, Any]
) -> Dict[str, Any]:
    """生成多次运行配置（评估稳定性）"""
    
    experiments = []
    for seed in seeds:
        exp = base_config.copy()
        exp['dataset'] = dataset
        exp['model'] = model
        exp['seed'] = seed
        experiments.append(exp)
    
    return {
        "description": f"{model} 在 {dataset} 上的稳定性测试 ({len(seeds)} 次运行)",
        "experiments": experiments
    }


def main():
    parser = argparse.ArgumentParser(
        description='实验配置生成工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

1. 模型对比（在MUTAG上测试所有模型）:
   python generate_config.py --type model_comparison --dataset MUTAG --output model_comp.json

2. 数据集对比（用GCN在多个数据集上测试）:
   python generate_config.py --type dataset_comparison --model gcn \\
       --datasets MUTAG,PROTEINS,DD --output dataset_comp.json

3. 超参数搜索:
   python generate_config.py --type hyperparam_search --dataset MUTAG --model gcn \\
       --hidden_dims 32,64,128 --dropouts 0.3,0.5,0.7 --lrs 0.001,0.01 \\
       --output hyperparam.json

4. 稳定性测试（多个随机种子）:
   python generate_config.py --type seed_robustness --dataset MUTAG --model gcn \\
       --seeds 42,123,456,789,1024 --output robustness.json
        """
    )
    
    parser.add_argument('--type', type=str, required=True,
                        choices=['model_comparison', 'dataset_comparison', 
                                'hyperparam_search', 'seed_robustness'],
                        help='配置类型')
    
    parser.add_argument('--dataset', type=str, default='MUTAG',
                        help='数据集名称（单个）')
    parser.add_argument('--datasets', type=str, default='MUTAG,PROTEINS,DD',
                        help='数据集列表（逗号分隔）')
    
    parser.add_argument('--model', type=str, default='gcn',
                        help='模型名称（单个）')
    parser.add_argument('--models', type=str, 
                        default='gcn,gat,gin,sage,gatv2,chebnet',
                        help='模型列表（逗号分隔）')
    
    # 超参数选项
    parser.add_argument('--hidden_dims', type=str, default='32,64,128',
                        help='隐藏层维度列表（逗号分隔）')
    parser.add_argument('--dropouts', type=str, default='0.3,0.5,0.7',
                        help='Dropout比例列表（逗号分隔）')
    parser.add_argument('--lrs', type=str, default='0.001,0.01',
                        help='学习率列表（逗号分隔）')
    parser.add_argument('--seeds', type=str, default='42,123,456,789,1024',
                        help='随机种子列表（逗号分隔）')
    
    # 基础配置
    parser.add_argument('--data_source', type=str, default='tudataset',
                        help='数据源类型')
    parser.add_argument('--data_root', type=str, default='../../data/TUDataset',
                        help='数据集根目录')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批大小')
    parser.add_argument('--epochs', type=int, default=200,
                        help='训练轮数')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping耐心值')
    
    parser.add_argument('--output', type=str, default='experiments_config.json',
                        help='输出文件路径')
    
    args = parser.parse_args()
    
    # 基础配置
    base_config = {
        'data_source': args.data_source,
        'data_root': args.data_root,
        'hidden_dim': 64,
        'dropout': 0.5,
        'batch_size': args.batch_size,
        'lr': 0.001,
        'weight_decay': 0.0005,
        'epochs': args.epochs,
        'patience': args.patience,
        'seed': 42,
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'print_every': 10
    }
    
    # 根据类型生成配置
    if args.type == 'model_comparison':
        models = [m.strip() for m in args.models.split(',')]
        config = generate_model_comparison_config(args.dataset, models, base_config)
    
    elif args.type == 'dataset_comparison':
        datasets = [d.strip() for d in args.datasets.split(',')]
        config = generate_dataset_comparison_config(datasets, args.model, base_config)
    
    elif args.type == 'hyperparam_search':
        hidden_dims = [int(h) for h in args.hidden_dims.split(',')]
        dropouts = [float(d) for d in args.dropouts.split(',')]
        lrs = [float(l) for l in args.lrs.split(',')]
        config = generate_hyperparam_search_config(
            args.dataset, args.model, hidden_dims, dropouts, lrs, base_config
        )
    
    elif args.type == 'seed_robustness':
        seeds = [int(s) for s in args.seeds.split(',')]
        config = generate_seed_robustness_config(
            args.dataset, args.model, seeds, base_config
        )
    
    # 保存配置
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    
    print(f"✅ 配置文件已生成: {args.output}")
    print(f"\n配置信息:")
    print(f"  - 类型: {args.type}")
    print(f"  - 实验数量: {len(config['experiments'])}")
    print(f"  - 描述: {config['description']}")
    print(f"\n运行实验:")
    print(f"  python run.py --config {args.output}")


if __name__ == '__main__':
    main()

