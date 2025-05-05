# bettergcn包初始化文件
"""
神经元网络图卷积网络增强模块

此包包含用于改进GCN的模型、数据处理和图构建功能
"""

__version__ = '0.1.0'

# 导入模型
from .model import ImprovedGCN

# 导入数据处理和图构建函数
from .process import (
    load_data, 
    generate_graph, 
    apply_smote, 
    create_dataset,
    split_data,
    knn_graph
)

__all__ = [
    'ImprovedGCN', 
    'load_data', 
    'generate_graph', 
    'apply_smote', 
    'create_dataset',
    'split_data',
    'knn_graph'
] 