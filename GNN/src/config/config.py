"""
配置模块

该模块定义了GNN模型训练和可视化的配置参数
"""

import os
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

class Config:
    """
    神经元GNN分析配置类
    
    用于存储和管理GNN模型训练和可视化的配置参数
    """
    
    def __init__(self):
        """
        初始化配置参数
        """
        # 项目路径
        self.base_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
        self.data_dir = self.base_dir / 'datasets'
        self.models_dir = self.base_dir / 'models'
        self.results_dir = self.base_dir / 'results'
        
        # 确保目录存在
        self.models_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # 数据文件路径
        self.day6_data_path = self.data_dir / 'Day6_with_behavior_labels_filled.xlsx'
        
        # 模型参数
        self.model_type = 'gcn'  # 可选: 'gcn', 'gat'
        self.hidden_channels = 64
        self.out_channels = 32
        self.num_layers = 2
        self.dropout = 0.2
        self.learning_rate = 0.001
        self.weight_decay = 5e-4
        self.epochs = 100
        self.patience = 10  # 早停参数
        
        # 图构建参数
        self.correlation_threshold = 0.5  # 相关性阈值，用于构建边
        self.edge_weight_method = 'correlation'  # 边权重计算方法
        
        # 可视化参数
        self.node_size = 300
        self.edge_width = 2
        self.background_opacity = 0.3
        self.color_scheme = 'tab20'
        self.max_groups = 8
        
    def get_model_path(self, model_name: str) -> Path:
        """
        获取模型保存路径
        
        参数:
            model_name: 模型名称
            
        返回:
            模型保存的完整路径
        """
        return self.models_dir / f"{model_name}.pt"
    
    def get_results_path(self, result_name: str) -> Path:
        """
        获取结果保存路径
        
        参数:
            result_name: 结果文件名
            
        返回:
            结果文件的完整路径
        """
        return self.results_dir / result_name
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将配置参数转换为字典
        
        返回:
            包含所有配置参数的字典
        """
        return {
            'model_type': self.model_type,
            'hidden_channels': self.hidden_channels, 
            'out_channels': self.out_channels,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'epochs': self.epochs,
            'patience': self.patience,
            'correlation_threshold': self.correlation_threshold,
            'edge_weight_method': self.edge_weight_method
        }

# 创建全局配置实例
config = Config() 