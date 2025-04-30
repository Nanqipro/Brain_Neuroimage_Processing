"""
神经元活动归因分析包初始化文件

该模块导入并暴露了归因分析项目中的关键组件：
- CNN和LSTM模型：用于神经元活动的时间序列分类
- NeuronData：用于加载和处理神经元活动数据集

作者: SCN研究小组
日期: 2023
"""

from .model import CNN, LSTM           # 导入模型定义
from .dataset import NeuronData        # 导入数据集类