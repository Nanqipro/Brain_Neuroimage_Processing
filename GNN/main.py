"""
神经元拓扑结构分析主程序

该程序使用GNN模型分析神经元钙离子浓度波动数据，构建神经元拓扑结构
"""

import torch
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os

from src.config.config import Config
from src.utils.data_processor import NeuronDataProcessor
from src.core.gcn_model import NeuronGCN
from src.core.gat_model import NeuronGAT
from src.core.train import ModelTrainer
from src.utils.visualize import TopologyVisualizer

def parse_args():
    """
    解析命令行参数
    
    返回:
        解析后的参数
    """
    parser = argparse.ArgumentParser(description='神经元拓扑结构分析')
    
    parser.add_argument('--model_type', type=str, default='gcn', choices=['gcn', 'gat'],
                        help='模型类型: gcn或gat')
    parser.add_argument('--hidden_channels', type=int, default=64,
                        help='隐藏层维度')
    parser.add_argument('--out_channels', type=int, default=32,
                        help='输出特征维度')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='GNN层数')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout比例')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--correlation_threshold', type=float, default=0.5,
                        help='相关性阈值')
    parser.add_argument('--train', action='store_true',
                        help='是否训练模型')
    parser.add_argument('--visualize', action='store_true',
                        help='是否可视化拓扑结构')
    parser.add_argument('--analyze', action='store_true',
                        help='是否分析社区结构')
    
    return parser.parse_args()

def main():
    """
    主函数
    """
    # 解析命令行参数
    args = parse_args()
    
    # 创建配置对象
    config = Config()
    
    # 更新配置
    config.model_type = args.model_type
    config.hidden_channels = args.hidden_channels
    config.out_channels = args.out_channels
    config.num_layers = args.num_layers
    config.dropout = args.dropout
    config.learning_rate = args.lr
    config.epochs = args.epochs
    config.correlation_threshold = args.correlation_threshold
    
    print(f"使用模型: {config.model_type.upper()}")
    
    # 打印设备信息
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 初始化数据处理器
    data_processor = NeuronDataProcessor(config)
    
    # 加载数据
    data_processor.load_data()
    
    # 构建图
    G = data_processor.build_graph()
    
    # 可视化原始图
    data_processor.visualize_graph(config.get_results_path("original_graph.png"))
    
    # 转换为PyTorch Geometric数据
    data = data_processor.to_pytorch_geometric()
    
    # 创建模型
    if config.model_type == 'gcn':
        model = NeuronGCN(
            in_channels=data.x.shape[1],
            hidden_channels=config.hidden_channels,
            out_channels=config.out_channels,
            num_layers=config.num_layers,
            dropout=config.dropout
        )
    else:  # 'gat'
        model = NeuronGAT(
            in_channels=data.x.shape[1],
            hidden_channels=config.hidden_channels,
            out_channels=config.out_channels,
            num_layers=config.num_layers,
            heads=4,  # 注意力头数
            dropout=config.dropout
        )
    
    # 初始化训练器
    trainer = ModelTrainer(model, config, device)
    
    # 训练或加载模型
    if args.train:
        # 训练模型
        trainer.train(data)
        # 保存模型
        trainer.save_model()
        # 绘制训练历史
        trainer.plot_training_history()
    else:
        # 尝试加载模型
        try:
            trainer.load_model()
            print("成功加载预训练模型")
        except FileNotFoundError:
            print("没有找到预训练模型，将进行训练...")
            # 训练模型
            trainer.train(data)
            # 保存模型
            trainer.save_model()
            # 绘制训练历史
            trainer.plot_training_history()
    
    # 提取神经元嵌入
    embedding_dict, similarities = trainer.extract_neuron_embeddings(data, data_processor.neuron_columns)
    
    # 可视化嵌入
    trainer.visualize_embeddings(embedding_dict, data_processor.neuron_columns)
    
    # 创建拓扑结构
    G_gnn = trainer.create_gnn_topology(embedding_dict, data_processor.neuron_columns, threshold=0.7)
    
    # 可视化和分析
    if args.visualize or args.analyze:
        # 初始化可视化器
        visualizer = TopologyVisualizer(config)
        
        # 可视化拓扑结构
        if args.visualize:
            visualizer.visualize_topology_static(G_gnn)
            visualizer.visualize_topology_interactive(G_gnn)
        
        # 分析社区结构
        if args.analyze:
            analysis_result = visualizer.analyze_communities(G_gnn)
            
            # 打印社区分析结果摘要
            print("\n社区分析结果摘要:")
            print(f"总节点数: {analysis_result['global']['num_nodes']}")
            print(f"总边数: {analysis_result['global']['num_edges']}")
            print(f"社区数量: {analysis_result['global']['num_communities']}")
            print(f"图密度: {analysis_result['global']['density']:.4f}")
            print(f"平均聚类系数: {analysis_result['global']['avg_clustering']:.4f}")
            print(f"模块度: {analysis_result['global']['modularity']:.4f}")
            
            # 打印各社区信息
            print("\n各社区信息:")
            for community in analysis_result['communities']:
                print(f"社区 {community['id']}:")
                print(f"  大小: {community['size']} 个节点")
                print(f"  中心节点: {community['central_node']}")
                print(f"  内部密度: {community['internal_density']:.4f}")
                print(f"  直径: {community['diameter']}")
                print(f"  节点: {', '.join(community['nodes'][:5])}{'...' if len(community['nodes']) > 5 else ''}")
                print("")
    
    print("\n神经元拓扑结构分析完成！")

if __name__ == "__main__":
    main() 