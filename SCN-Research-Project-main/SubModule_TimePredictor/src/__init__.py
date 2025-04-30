"""
子模块时间预测器包 - 用于不同空间区域神经元时间序列分析的工具包

包含学习率调度器、CNN模型以及数据加载工具等核心组件，
专用于在特定脑区子模块上进行训练和交叉测试
"""

from .lr_updater import CosineLrUpdater
from .model import CNN
from .dataset import TestDataset, TrainDataset