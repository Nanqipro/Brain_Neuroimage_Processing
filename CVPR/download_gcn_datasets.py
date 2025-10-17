#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GCN标准数据集下载脚本

支持从多个来源下载常用的GCN数据集：
1. Cora - 论文引用网络（2708节点，7类别）
2. Citeseer - 论文引用网络（3327节点，6类别）
3. Pubmed - 论文引用网络（19717节点，3类别）
4. Reddit - 大规模社交网络
5. PPI - 蛋白质相互作用网络

使用方法:
    python download_gcn_datasets.py --datasets cora citeseer pubmed --method pyg
    python download_gcn_datasets.py --datasets all --method modelscope
"""

import os
import argparse
import sys
from pathlib import Path


def download_from_pyg(dataset_name, data_root):
    """
    使用PyTorch Geometric下载数据集
    """
    try:
        from torch_geometric.datasets import Planetoid, Reddit, PPI
        print(f"正在使用PyTorch Geometric下载 {dataset_name}...")
        
        dataset_lower = dataset_name.lower()
        
        if dataset_lower in ['cora', 'citeseer', 'pubmed']:
            dataset = Planetoid(root=data_root, name=dataset_name)
            print(f"✓ {dataset_name} 下载完成!")
            print(f"  - 节点数: {dataset[0].num_nodes}")
            print(f"  - 边数: {dataset[0].num_edges}")
            print(f"  - 特征维度: {dataset.num_features}")
            print(f"  - 类别数: {dataset.num_classes}")
            print(f"  - 保存路径: {os.path.join(data_root, dataset_name)}")
            
        elif dataset_lower == 'reddit':
            dataset = Reddit(root=os.path.join(data_root, 'Reddit'))
            print(f"✓ Reddit 下载完成!")
            print(f"  - 节点数: {dataset[0].num_nodes}")
            print(f"  - 边数: {dataset[0].num_edges}")
            print(f"  - 特征维度: {dataset.num_features}")
            print(f"  - 类别数: {dataset.num_classes}")
            print(f"  - 保存路径: {os.path.join(data_root, 'Reddit')}")
            
        elif dataset_lower == 'ppi':
            train_dataset = PPI(root=os.path.join(data_root, 'PPI'), split='train')
            val_dataset = PPI(root=os.path.join(data_root, 'PPI'), split='val')
            test_dataset = PPI(root=os.path.join(data_root, 'PPI'), split='test')
            print(f"✓ PPI 下载完成!")
            print(f"  - 训练图数: {len(train_dataset)}")
            print(f"  - 验证图数: {len(val_dataset)}")
            print(f"  - 测试图数: {len(test_dataset)}")
            print(f"  - 特征维度: {train_dataset.num_features}")
            print(f"  - 类别数: {train_dataset.num_classes}")
            print(f"  - 保存路径: {os.path.join(data_root, 'PPI')}")
        else:
            print(f"✗ 不支持的数据集: {dataset_name}")
            return False
            
        return True
        
    except ImportError:
        print("错误: 未安装 torch_geometric")
        print("请运行: pip install torch-geometric")
        return False
    except Exception as e:
        print(f"✗ 下载 {dataset_name} 时出错: {str(e)}")
        return False


def download_from_dgl(dataset_name, data_root):
    """
    使用DGL下载数据集
    """
    try:
        import dgl
        from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
        from dgl.data import RedditDataset, PPIDataset
        
        print(f"正在使用DGL下载 {dataset_name}...")
        
        dataset_lower = dataset_name.lower()
        
        if dataset_lower == 'cora':
            dataset = CoraGraphDataset(raw_dir=data_root)
        elif dataset_lower == 'citeseer':
            dataset = CiteseerGraphDataset(raw_dir=data_root)
        elif dataset_lower == 'pubmed':
            dataset = PubmedGraphDataset(raw_dir=data_root)
        elif dataset_lower == 'reddit':
            dataset = RedditDataset(raw_dir=data_root)
        elif dataset_lower == 'ppi':
            train_dataset = PPIDataset(mode='train', raw_dir=data_root)
            val_dataset = PPIDataset(mode='valid', raw_dir=data_root)
            test_dataset = PPIDataset(mode='test', raw_dir=data_root)
            print(f"✓ PPI 下载完成!")
            print(f"  - 训练图数: {len(train_dataset)}")
            print(f"  - 验证图数: {len(val_dataset)}")
            print(f"  - 测试图数: {len(test_dataset)}")
            print(f"  - 保存路径: {data_root}")
            return True
        else:
            print(f"✗ 不支持的数据集: {dataset_name}")
            return False
        
        if dataset_lower in ['cora', 'citeseer', 'pubmed']:
            graph = dataset[0]
            print(f"✓ {dataset_name} 下载完成!")
            print(f"  - 节点数: {graph.num_nodes()}")
            print(f"  - 边数: {graph.num_edges()}")
            print(f"  - 特征维度: {graph.ndata['feat'].shape[1]}")
            print(f"  - 类别数: {dataset.num_classes}")
        elif dataset_lower == 'reddit':
            graph = dataset[0]
            print(f"✓ Reddit 下载完成!")
            print(f"  - 节点数: {graph.num_nodes()}")
            print(f"  - 边数: {graph.num_edges()}")
            print(f"  - 特征维度: {graph.ndata['feat'].shape[1]}")
            print(f"  - 类别数: {dataset.num_classes}")
        
        print(f"  - 保存路径: {data_root}")
        return True
        
    except ImportError:
        print("错误: 未安装 dgl")
        print("请运行: pip install dgl")
        return False
    except Exception as e:
        print(f"✗ 下载 {dataset_name} 时出错: {str(e)}")
        return False


def download_from_modelscope(dataset_name, data_root):
    """
    尝试从ModelScope下载数据集
    """
    try:
        from modelscope.msdatasets import MsDataset
        
        print(f"正在尝试从ModelScope下载 {dataset_name}...")
        
        # ModelScope的数据集命名可能不同，这里提供一些可能的映射
        modelscope_mapping = {
            'cora': 'cora-dataset',  # 需要确认实际的数据集ID
            'citeseer': 'citeseer-dataset',
            'pubmed': 'pubmed-dataset',
        }
        
        ms_name = modelscope_mapping.get(dataset_name.lower(), dataset_name.lower())
        
        try:
            # 尝试加载数据集
            ds = MsDataset.load(ms_name, cache_dir=data_root)
            print(f"✓ {dataset_name} 从ModelScope下载完成!")
            print(f"  - 保存路径: {data_root}")
            return True
        except Exception as e:
            print(f"✗ ModelScope中未找到 {dataset_name} 数据集")
            print(f"  提示: {str(e)}")
            print(f"  建议使用 --method pyg 或 --method dgl 下载标准数据集")
            return False
            
    except ImportError:
        print("错误: 未安装 modelscope")
        print("请运行: pip install modelscope")
        return False


def main():
    parser = argparse.ArgumentParser(description='下载常用的GCN数据集')
    parser.add_argument('--datasets', nargs='+', 
                        choices=['cora', 'citeseer', 'pubmed', 'reddit', 'ppi', 'all'],
                        default=['cora'],
                        help='要下载的数据集名称')
    parser.add_argument('--method', 
                        choices=['pyg', 'dgl', 'modelscope'],
                        default='pyg',
                        help='下载方法: pyg(PyTorch Geometric), dgl(Deep Graph Library), modelscope')
    parser.add_argument('--data_root', 
                        type=str,
                        default='./data/gcn_datasets',
                        help='数据集保存路径')
    
    args = parser.parse_args()
    
    # 如果选择all，则下载所有数据集
    if 'all' in args.datasets:
        args.datasets = ['cora', 'citeseer', 'pubmed', 'reddit', 'ppi']
    
    # 创建数据目录
    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    data_root = str(data_root.absolute())
    
    print("=" * 60)
    print("GCN 标准数据集下载工具")
    print("=" * 60)
    print(f"数据集: {', '.join(args.datasets)}")
    print(f"下载方法: {args.method}")
    print(f"保存路径: {data_root}")
    print("=" * 60)
    print()
    
    # 选择下载方法
    if args.method == 'pyg':
        download_func = download_from_pyg
    elif args.method == 'dgl':
        download_func = download_from_dgl
    elif args.method == 'modelscope':
        download_func = download_from_modelscope
    else:
        print(f"不支持的下载方法: {args.method}")
        sys.exit(1)
    
    # 下载数据集
    success_count = 0
    for dataset_name in args.datasets:
        print(f"\n{'=' * 60}")
        if download_func(dataset_name, data_root):
            success_count += 1
        print()
    
    print("=" * 60)
    print(f"下载完成! 成功: {success_count}/{len(args.datasets)}")
    print("=" * 60)
    
    # 打印数据集使用示例
    if success_count > 0:
        print("\n使用示例:")
        print("=" * 60)
        if args.method == 'pyg':
            print("from torch_geometric.datasets import Planetoid")
            print(f"dataset = Planetoid(root='{data_root}', name='Cora')")
            print("data = dataset[0]")
        elif args.method == 'dgl':
            print("from dgl.data import CoraGraphDataset")
            print(f"dataset = CoraGraphDataset(raw_dir='{data_root}')")
            print("graph = dataset[0]")
        print("=" * 60)


if __name__ == "__main__":
    main()

