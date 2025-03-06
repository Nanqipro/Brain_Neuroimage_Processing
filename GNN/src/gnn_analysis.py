import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import torch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

# 导入我们的GNN模型
import sys
sys.path.append('.')  # 确保能够导入当前目录的模块
from gnn_models import BrainGNN, NeuronGNNLSTM, BrainNodeFeatureExtractor
from gnn_models import convert_networkx_to_pyg, create_dynamic_graph_data

# 设置随机种子确保可重复性
def set_random_seed(seed):
    """设置随机种子以确保结果可重复"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DynamicGraphAnalyzer:
    """神经元动态图分析器
    
    使用GNN模型分析神经元网络的动态连接模式
    """
    def __init__(self, config):
        """
        初始化动态图分析器
        
        参数:
            config: 配置对象，包含数据路径和模型参数
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 设置随机种子
        set_random_seed(config.random_seed)
        
        # 创建结果目录
        os.makedirs(config.output_dir, exist_ok=True)