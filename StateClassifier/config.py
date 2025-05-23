#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
脑网络状态分类器配置文件

集中管理项目中的所有配置参数，包括数据处理、模型训练和文件路径等设置。

作者: Clade 4
日期: 2025年5月23日
"""

import os
from pathlib import Path

class Config:
    """
    项目配置类
    
    包含所有项目相关的配置参数，便于统一管理和修改。
    """
    
    # ==================== 路径配置 ====================
    
    # 项目根目录
    PROJECT_ROOT = Path(__file__).parent
    
    # 数据目录
    DATA_DIR = PROJECT_ROOT / "datasets"
    RESULT_DIR = PROJECT_ROOT / "results"
    SRC_DIR = PROJECT_ROOT / "src"
    
    # 输入数据文件
    INPUT_DATA_PATH = "datasets/EMtrace01.xlsx"
    
    # 输出文件名
    NODES_CSV = "nodes.csv"
    EDGES_CSV = "edges.csv"
    GRAPHS_CSV = "graphs.csv"
    
    # 模型保存路径
    BEST_MODEL_PATH = RESULT_DIR / "best_scn.pt"
    
    # ==================== 数据处理配置 ====================
    
    # 钙成像参数
    FRAME_RATE = 0.67  # 帧率(Hz)
    
    # 相空间重构参数
    EMBEDDING_DIM = 3  # 嵌入维度(3D相空间)
    TRAJECTORY_LENGTH = 170  # 统一轨迹长度
    MAX_DELAY = 50  # 互信息计算的最大延迟
    NUM_BINS = 16  # 互信息计算的分箱数量
    
    # 数据标准化参数
    ZSCORE_AXIS = None  # Z-score标准化的轴向(None表示全局标准化)
    
    # 数据裁剪参数
    MAX_CELLS_PROCESS = 10  # 最大处理细胞数量
    
    # ==================== 模型配置 ====================
    
    # 网络架构参数
    INPUT_FEATURES = 3  # 输入特征维度(x,y,z坐标)
    HIDDEN_DIM = 32  # 隐藏层维度
    NUM_GCN_LAYERS = 3  # GCN层数
    NUM_CLASSES = 6  # 分类类别数
    DROPOUT_RATE = 0.5  # Dropout比率
    
    # 池化参数
    POOLING_TYPE = "global"  # 池化类型: "global", "max", "mean"
    
    # ==================== 训练配置 ====================
    
    # 训练参数
    NUM_EPOCHS = 160  # 训练轮数
    LEARNING_RATE = 0.001  # 初始学习率
    WEIGHT_DECAY = 1e-4  # L2正则化系数
    BATCH_SIZE = 1  # 批次大小
    
    # 学习率调度
    LR_DECAY_FACTOR = 0.75  # 学习率衰减因子
    LR_DECAY_STEP = 20  # 学习率衰减步长
    
    # 数据集分割比例
    TRAIN_RATIO = 0.6  # 训练集比例
    VALID_RATIO = 0.2  # 验证集比例
    TEST_RATIO = 0.2  # 测试集比例
    
    # 数据增强参数
    AUGMENTATION_FACTOR = 4  # 数据增强倍数
    NOISE_STD = 0.5  # 噪声标准差
    
    # ==================== 设备配置 ====================
    
    # 计算设备
    DEVICE = "cuda"  # "cuda" 或 "cpu"
    
    # 随机种子
    RANDOM_SEED = 42
    TORCH_SEED = 0
    
    # ==================== 可视化配置 ====================
    
    # 图形参数
    FIGURE_SIZE = (10, 8)  # 图形尺寸
    DPI = 300  # 图形分辨率
    
    # 相空间可视化参数
    PLOT_3D = True  # 是否显示3D图
    POINT_SIZE = 1  # 散点大小
    ALPHA = 0.7  # 透明度
    
    # ==================== 日志配置 ====================
    
    # 日志参数
    LOG_LEVEL = "INFO"  # 日志级别
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # 进度条配置
    PROGRESS_BAR = True  # 是否显示进度条
    
    # ==================== 验证和错误处理 ====================
    
    # 数据验证参数
    MIN_SIGNAL_LENGTH = 50  # 最小信号长度
    MAX_SIGNAL_LENGTH = 1000  # 最大信号长度
    
    # 错误处理
    CONTINUE_ON_ERROR = True  # 遇到错误时是否继续处理
    
    @classmethod
    def create_directories(cls):
        """
        创建必要的目录结构
        """
        directories = [cls.DATA_DIR, cls.RESULT_DIR]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def validate_config(cls):
        """
        验证配置参数的有效性
        
        Returns
        -------
        bool
            配置是否有效
        """
        # 检查比例参数
        if abs(cls.TRAIN_RATIO + cls.VALID_RATIO + cls.TEST_RATIO - 1.0) > 1e-6:
            raise ValueError("训练、验证和测试集比例之和必须等于1")
        
        # 检查正数参数
        positive_params = [
            cls.FRAME_RATE, cls.EMBEDDING_DIM, cls.TRAJECTORY_LENGTH,
            cls.NUM_EPOCHS, cls.LEARNING_RATE, cls.BATCH_SIZE
        ]
        
        for param in positive_params:
            if param <= 0:
                raise ValueError(f"参数必须为正数: {param}")
        
        # 检查范围参数
        if not 0 <= cls.DROPOUT_RATE <= 1:
            raise ValueError("Dropout率必须在0-1之间")
        
        return True
    
    @classmethod
    def get_config_dict(cls):
        """
        获取配置字典，用于保存和记录
        
        Returns
        -------
        dict
            包含所有配置参数的字典
        """
        config_dict = {}
        for attr_name in dir(cls):
            if not attr_name.startswith('_') and not callable(getattr(cls, attr_name)):
                config_dict[attr_name] = getattr(cls, attr_name)
        
        return config_dict


# 全局配置实例
config = Config()

# 验证配置并创建目录
if __name__ == "__main__":
    try:
        config.validate_config()
        config.create_directories()
        print("✓ 配置验证通过，目录创建成功")
        
        # 打印主要配置参数
        print("\n主要配置参数:")
        print(f"  - 嵌入维度: {config.EMBEDDING_DIM}")
        print(f"  - 轨迹长度: {config.TRAJECTORY_LENGTH}")
        print(f"  - 训练轮数: {config.NUM_EPOCHS}")
        print(f"  - 学习率: {config.LEARNING_RATE}")
        print(f"  - 分类类别: {config.NUM_CLASSES}")
        
    except Exception as e:
        print(f"✗ 配置验证失败: {e}") 