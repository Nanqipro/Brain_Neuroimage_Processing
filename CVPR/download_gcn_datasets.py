#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GCN数据集下载脚本

支持两种类型的数据集：

【图分类数据集】（推荐！适合您的代码 - 使用DataLoader + global_pooling）
- TUDataset: MUTAG, PROTEINS, NCI1, ENZYMES, IMDB-BINARY 等
- OGB: ogbg-molhiv, ogbg-ppa 等

【节点分类数据集】（仅供参考，需要不同的代码结构）
- Planetoid: Cora, CiteSeer, Pubmed
- Reddit, PPI

使用方法:
    # 查看所有可用数据集
    python download_gcn_datasets.py --list
    
    # 下载图分类数据集（推荐）
    python download_gcn_datasets.py --dataset MUTAG --task graph
    python download_gcn_datasets.py --dataset ogbg-molhiv --task graph --source ogb
    
    # 下载节点分类数据集
    python download_gcn_datasets.py --dataset cora --task node
"""

import os
import argparse
import sys
from pathlib import Path

# # -----------------------------------------------------------------
# # 解决 PyTorch 2.6+ 安全加载问题 (torch.load weights_only=True)
# # -----------------------------------------------------------------
# try:
#     # 导入 torch 和 DataEdgeAttr 所在的模块
#     import torch
#     import torch_geometric.data.data 
    
#     # 明确告诉 PyTorch 信任这个来自 torch_geometric 的类
#     torch.serialization.add_safe_globals([
#         torch_geometric.data.data.DataEdgeAttr,
#         torch_geometric.data.data.DataTensorAttr,
#         torch_geometric.data.storage.GlobalStorage
#     ])
# except ImportError:
#     # 如果用户还没安装 torch/torch_geometric，先跳过
#     # 后续的下载函数会捕获这个错误并提示安装
#     pass
# # -----------------------------------------------------------------


# ============ 图分类数据集信息 ============
# 这些数据集适合您的代码（使用 DataLoader + global_pooling）

TU_DATASETS = {
    # 小型数据集（快速测试）
    'MUTAG': {
        'graphs': 188,
        'classes': 2,
        'description': '分子图分类 - 判断化合物致癌性',
        'avg_nodes': 17.9,
        'task': '二分类',
        'domain': '生物化学'
    },
    'PROTEINS': {
        'graphs': 1113,
        'classes': 2,
        'description': '蛋白质图分类 - 判断蛋白质是否为酶',
        'avg_nodes': 39.1,
        'task': '二分类',
        'domain': '生物'
    },
    'PTC_MR': {
        'graphs': 344,
        'classes': 2,
        'description': '化合物致癌性预测',
        'avg_nodes': 14.3,
        'task': '二分类',
        'domain': '生物化学'
    },
    'ENZYMES': {
        'graphs': 600,
        'classes': 6,
        'description': '蛋白质图分类 - 酶类型分类',
        'avg_nodes': 32.6,
        'task': '多分类',
        'domain': '生物'
    },
    'NCI1': {
        'graphs': 4110,
        'classes': 2,
        'description': '化合物抗癌活性预测',
        'avg_nodes': 29.9,
        'task': '二分类',
        'domain': '生物化学'
    },
    'NCI109': {
        'graphs': 4127,
        'classes': 2,
        'description': '化合物抗癌活性预测',
        'avg_nodes': 29.7,
        'task': '二分类',
        'domain': '生物化学'
    },
    'IMDB-BINARY': {
        'graphs': 1000,
        'classes': 2,
        'description': '电影协作网络 - 判断电影类型',
        'avg_nodes': 19.8,
        'task': '二分类',
        'domain': '社交网络'
    },
    'IMDB-MULTI': {
        'graphs': 1500,
        'classes': 3,
        'description': '电影协作网络 - 判断电影类型（3类）',
        'avg_nodes': 13.0,
        'task': '多分类',
        'domain': '社交网络'
    },
    'REDDIT-BINARY': {
        'graphs': 2000,
        'classes': 2,
        'description': 'Reddit社区图分类',
        'avg_nodes': 429.6,
        'task': '二分类',
        'domain': '社交网络'
    },
    'DD': {
        'graphs': 1178,
        'classes': 2,
        'description': '蛋白质图分类 - 酶/非酶',
        'avg_nodes': 284.3,
        'task': '二分类',
        'domain': '生物'
    },
}

OGB_GRAPH_DATASETS = {
    'ogbg-molhiv': {
        'graphs': 41127,
        'classes': 2,
        'description': '分子图 - HIV抑制剂预测',
        'task': '二分类',
        'domain': '药物发现'
    },
    'ogbg-molpcba': {
        'graphs': 437929,
        'classes': 128,
        'description': '分子图 - 生物活性预测（大规模）',
        'task': '多标签分类',
        'domain': '药物发现'
    },
    'ogbg-ppa': {
        'graphs': 158100,
        'classes': 37,
        'description': '蛋白质-蛋白质相互作用图',
        'task': '多分类',
        'domain': '生物'
    },
}

# 节点分类数据集（需要不同的代码结构）
NODE_DATASETS = {
    'cora': '论文引用网络 (2708节点, 7类)',
    'citeseer': '论文引用网络 (3327节点, 6类)',
    'pubmed': '论文引用网络 (19717节点, 3类)',
    'reddit': '社交网络 (232965节点, 41类)',
    'ppi': '蛋白质网络 (24图, 多标签)'
}


def download_tu_dataset(dataset_name, data_root):
    """下载TUDataset图分类数据集（推荐！适合您的代码）"""
    try:
        from torch_geometric.datasets import TUDataset
        from torch_geometric.loader import DataLoader
        
        print(f"正在下载 TUDataset/{dataset_name}...")
        
        dataset = TUDataset(root=os.path.join(data_root, 'TUDataset'), name=dataset_name)
        
        print(f"✓ {dataset_name} 下载完成!")
        print(f"  - 图数量: {len(dataset)}")
        print(f"  - 类别数: {dataset.num_classes}")
        print(f"  - 特征维度: {dataset.num_features}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"  - 示例图节点数: {sample.num_nodes}")
            print(f"  - 示例图边数: {sample.num_edges}")
        
        print(f"  - 保存路径: {os.path.join(data_root, 'TUDataset', dataset_name)}")
        print(f"\n💡 使用示例（适配您的代码）:")
        print(f"from torch_geometric.datasets import TUDataset")
        print(f"from torch_geometric.loader import DataLoader")
        print(f"dataset = TUDataset(root='{os.path.join(data_root, 'TUDataset')}', name='{dataset_name}')")
        print(f"loader = DataLoader(dataset, batch_size=32, shuffle=True)")
        
        return True
        
    except ImportError:
        print("错误: 未安装 torch_geometric")
        print("请运行: pip install torch-geometric")
        return False
    except Exception as e:
        print(f"✗ 下载 {dataset_name} 时出错: {str(e)}")
        return False


def download_ogb_graph_dataset(dataset_name, data_root):
    """下载OGB图分类数据集（推荐！适合您的代码）"""
    try:
        from ogb.graphproppred import PygGraphPropPredDataset
        from torch_geometric.loader import DataLoader
        
        print(f"正在从OGB下载 {dataset_name}...")
        
        dataset = PygGraphPropPredDataset(name=dataset_name, root=os.path.join(data_root, 'OGB'))
        split_idx = dataset.get_idx_split()
        
        print(f"✓ {dataset_name} 下载完成!")
        print(f"  - 总图数: {len(dataset)}")
        print(f"  - 训练集: {len(split_idx['train'])}")
        print(f"  - 验证集: {len(split_idx['valid'])}")
        print(f"  - 测试集: {len(split_idx['test'])}")
        print(f"  - 特征维度: {dataset.num_features}")
        print(f"  - 任务类型: {dataset.task_type}")
        print(f"  - 保存路径: {os.path.join(data_root, 'OGB', dataset_name)}")
        print(f"\n💡 使用示例（适配您的代码）:")
        print(f"from ogb.graphproppred import PygGraphPropPredDataset")
        print(f"from torch_geometric.loader import DataLoader")
        print(f"dataset = PygGraphPropPredDataset(name='{dataset_name}', root='{os.path.join(data_root, 'OGB')}')")
        print(f"split_idx = dataset.get_idx_split()")
        print(f"train_loader = DataLoader(dataset[split_idx['train']], batch_size=32)")
        
        return True
        
    except ImportError:
        print("错误: 未安装 ogb")
        print("请运行: pip install ogb")
        return False
    except Exception as e:
        print(f"✗ 下载 {dataset_name} 时出错: {str(e)}")
        return False


def download_node_classification_dataset(dataset_name, data_root):
    """下载节点分类数据集（需要不同的代码结构）"""
    try:
        from torch_geometric.datasets import Planetoid, Reddit, PPI
        print(f"正在下载节点分类数据集 {dataset_name}...")
        print("⚠️  注意: 这是节点分类数据集，与您当前的图分类代码不兼容！")
        
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


def print_available_datasets():
    """打印所有可用数据集"""
    print("\n" + "=" * 100)
    print("📊 可用数据集列表")
    print("=" * 100)
    
    print("\n✅ 【图分类数据集】- 推荐！适合您的代码（使用 DataLoader + global_pooling）")
    print("=" * 100)
    
    print("\n🔬 TUDataset 系列:")
    print("-" * 100)
    print(f"{'数据集名称':<20} {'图数量':<10} {'类别':<8} {'领域':<12} {'任务':<12} {'描述'}")
    print("-" * 100)
    
    for name, info in TU_DATASETS.items():
        print(f"{name:<20} {info['graphs']:<10} {info['classes']:<8} {info['domain']:<12} {info['task']:<12} {info['description']}")
    
    print("\n🏆 OGB (Open Graph Benchmark):")
    print("-" * 100)
    print(f"{'数据集名称':<20} {'图数量':<10} {'任务':<18} {'领域':<12} {'描述'}")
    print("-" * 100)
    
    for name, info in OGB_GRAPH_DATASETS.items():
        print(f"{name:<20} {info['graphs']:<10} {info['task']:<18} {info['domain']:<12} {info['description']}")
    
    print("\n" + "=" * 100)
    print("⚠️  【节点分类数据集】- 需要不同的代码结构（不推荐用于当前代码）")
    print("=" * 100)
    for name, desc in NODE_DATASETS.items():
        print(f"  • {name:<15} - {desc}")
    
    print("\n" + "=" * 100)
    print("💡 推荐使用顺序（图分类）:")
    print("  1. 快速测试:   python download_gcn_datasets.py --dataset MUTAG --task graph")
    print("  2. 中等规模:   python download_gcn_datasets.py --dataset PROTEINS --task graph")
    print("  3. 标准基准:   python download_gcn_datasets.py --dataset NCI1 --task graph")
    print("  4. 大规模OGB:  python download_gcn_datasets.py --dataset ogbg-molhiv --task graph --source ogb")
    print("=" * 100 + "\n")


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
    parser = argparse.ArgumentParser(
        description='GCN数据集下载工具 - 支持图分类和节点分类数据集',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 查看所有可用数据集
  python download_gcn_datasets.py --list
  
  # 下载图分类数据集（推荐！适合您的代码）
  python download_gcn_datasets.py --dataset MUTAG --task graph
  python download_gcn_datasets.py --dataset PROTEINS --task graph
  python download_gcn_datasets.py --dataset ogbg-molhiv --task graph --source ogb
  
  # 下载节点分类数据集（需要不同代码结构）
  python download_gcn_datasets.py --dataset cora --task node
        """
    )
    
    parser.add_argument('--list', action='store_true',
                        help='列出所有可用数据集')
    parser.add_argument('--dataset', type=str,
                        help='要下载的数据集名称')
    parser.add_argument('--task', type=str,
                        choices=['graph', 'node'],
                        default='graph',
                        help='任务类型: graph (图分类，推荐), node (节点分类)')
    parser.add_argument('--source', type=str,
                        choices=['tu', 'ogb', 'pyg', 'dgl'],
                        default='tu',
                        help='数据源: tu (TUDataset), ogb (OGB), pyg (PyG节点分类), dgl (DGL)')
    parser.add_argument('--data_root', type=str,
                        default='./data',
                        help='数据集保存根目录')
    
    args = parser.parse_args()
    
    # 如果只是列出数据集
    if args.list:
        print_available_datasets()
        return
    
    # 检查是否指定了数据集
    if not args.dataset:
        print("❌ 错误: 请指定要下载的数据集名称")
        print("\n使用 --list 查看所有可用数据集")
        print("\n快速开始:")
        print("  python download_gcn_datasets.py --list")
        print("  python download_gcn_datasets.py --dataset MUTAG --task graph")
        sys.exit(1)
    
    # 创建数据目录
    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    data_root = str(data_root.absolute())
    
    print("=" * 100)
    print("GCN 数据集下载工具")
    print("=" * 100)
    print(f"数据集: {args.dataset}")
    print(f"任务类型: {'图分类 (Graph Classification)' if args.task == 'graph' else '节点分类 (Node Classification)'}")
    print(f"数据源: {args.source.upper()}")
    print(f"保存路径: {data_root}")
    print("=" * 100)
    print()
    
    # 根据任务类型和数据源选择下载函数
    success = False
    
    if args.task == 'graph':
        # 图分类数据集
        if args.source == 'tu':
            if args.dataset in TU_DATASETS:
                info = TU_DATASETS[args.dataset]
                print(f"📝 数据集信息:")
                print(f"  - 图数量: {info['graphs']}")
                print(f"  - 类别数: {info['classes']}")
                print(f"  - 任务类型: {info['task']}")
                print(f"  - 应用领域: {info['domain']}")
                print(f"  - 描述: {info['description']}")
                print()
            success = download_tu_dataset(args.dataset, data_root)
            
        elif args.source == 'ogb':
            if args.dataset in OGB_GRAPH_DATASETS:
                info = OGB_GRAPH_DATASETS[args.dataset]
                print(f"📝 数据集信息:")
                print(f"  - 图数量: {info['graphs']}")
                print(f"  - 任务类型: {info['task']}")
                print(f"  - 应用领域: {info['domain']}")
                print(f"  - 描述: {info['description']}")
                print()
            success = download_ogb_graph_dataset(args.dataset, data_root)
        else:
            print(f"❌ 图分类任务不支持数据源: {args.source}")
            print("请使用 --source tu 或 --source ogb")
            sys.exit(1)
            
    elif args.task == 'node':
        # 节点分类数据集
        print("⚠️  警告: 您正在下载节点分类数据集")
        print("   这与您当前的图分类代码（使用global_pooling）不兼容！")
        print("   如果要测试图分类模型，请使用: --task graph\n")
        
        if args.source in ['pyg', 'dgl']:
            if args.source == 'pyg':
                success = download_node_classification_dataset(args.dataset, data_root)
            elif args.source == 'dgl':
                success = download_from_dgl(args.dataset, data_root)
        else:
            print(f"❌ 节点分类任务不支持数据源: {args.source}")
            print("请使用 --source pyg 或 --source dgl")
            sys.exit(1)
    
    # 结果总结
    print("\n" + "=" * 100)
    if success:
        print("✅ 下载成功！")
        print("=" * 100)
        
        if args.task == 'graph':
            print("\n💡 下一步: 在您的代码中使用数据集")
            print("-" * 100)
            print("示例代码:")
            if args.source == 'tu':
                print(f"""
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from model import ImprovedGCN  # 您的模型

# 加载数据集
dataset = TUDataset(root='{data_root}/TUDataset', name='{args.dataset}')

# 划分训练/测试集
train_size = int(len(dataset) * 0.8)
train_dataset = dataset[:train_size]
test_dataset = dataset[train_size:]

# 创建DataLoader（与您的代码兼容）
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# 初始化模型
model = ImprovedGCN(
    num_features=dataset.num_features,
    hidden_dim=64,
    num_classes=dataset.num_classes,
    dropout=0.3
)

# 开始训练（使用您的 train.py）
# ...
                """)
            elif args.source == 'ogb':
                print(f"""
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader
from model import ImprovedGCN  # 您的模型

# 加载数据集（OGB自带划分）
dataset = PygGraphPropPredDataset(name='{args.dataset}', root='{data_root}/OGB')
split_idx = dataset.get_idx_split()

# 创建DataLoader
train_loader = DataLoader(dataset[split_idx['train']], batch_size=32, shuffle=True)
val_loader = DataLoader(dataset[split_idx['valid']], batch_size=32)
test_loader = DataLoader(dataset[split_idx['test']], batch_size=32)

# 初始化模型
model = ImprovedGCN(
    num_features=dataset.num_features,
    hidden_dim=64,
    num_classes=dataset.num_tasks,  # OGB使用num_tasks
    dropout=0.3
)
                """)
    else:
        print("❌ 下载失败")
        print("=" * 100)
        print("\n请检查:")
        print("  1. 网络连接是否正常")
        print("  2. 是否安装了所需的库 (torch-geometric, ogb)")
        print("  3. 数据集名称是否正确")
    
    print("=" * 100)


if __name__ == "__main__":
    main()

