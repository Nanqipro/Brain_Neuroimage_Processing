import os
from typing import Dict, List, Optional, Union, Any

class AnalysisConfig:
    """
    分析配置类：管理神经元行为分析项目的所有配置参数
    
    该类负责集中管理项目中的所有配置信息，包括：
    1. 文件路径配置 - 数据文件、输出目录、模型保存位置等
    2. 模型参数配置 - LSTM、GNN等模型的超参数设置
    3. 分析参数配置 - 聚类、拓扑分析、时间序列分析等参数
    4. 可视化参数配置 - 图表尺寸、颜色方案、节点样式等
    5. 目录结构管理 - 自动创建和验证必要的目录结构
    
    该类确保所有组件使用一致的配置，并提供集中的参数调整接口。
    """
    def __init__(self) -> None:
        """
        初始化分析配置对象
        
        创建并设置所有必要的配置参数，包括路径、模型参数、分析参数和可视化参数。
        同时创建必要的目录结构并验证关键路径的有效性。
        """
        # 基础目录配置
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录
        
        # 数据路径配置
        self.data_dir = os.path.join(self.base_dir, 'datasets')  # 数据集目录
        self.data_file = os.path.join(self.data_dir, 'Day3_with_behavior_labels_filled.xlsx')  # 原始数据文件
        self.data_identifier = 'Day3'  # 从数据文件名提取标识符
        
        # 输出目录配置
        self.output_dir = os.path.join(self.base_dir, 'results')  # 结果输出总目录
        self.model_dir = os.path.join(self.base_dir, 'models')    # 模型保存目录
        self.analysis_dir = os.path.join(self.output_dir, f'analysis_{self.data_identifier}')  # 分析结果目录
        self.train_dir = os.path.join(self.output_dir, f'train_{self.data_identifier}')   # 训练结果目录
        self.log_file = os.path.join(self.analysis_dir, f'analysis_log_{self.data_identifier}.txt')  # 分析日志文件
        
        # 拓扑分析相关路径配置
        self.topology_dir = os.path.join(self.output_dir, f'topology_{self.data_identifier}')  # 拓扑分析结果目录
        self.topology_html = os.path.join(self.topology_dir, f'pos_topology_{self.data_identifier}.html')  # HTML动画输出路径
        self.topology_gif = os.path.join(self.topology_dir, f'pos_topology_{self.data_identifier}.gif')    # GIF动画输出路径
        
        # GNN分析相关路径配置
        self.gnn_results_dir = os.path.join(self.analysis_dir, 'gnn_results')  # GNN结果目录
        self.interactive_dir = os.path.join(self.analysis_dir, 'interactive')  # 交互式可视化目录
        
        # GNN交互式可视化文件路径
        self.gnn_interactive_template = os.path.join(self.gnn_results_dir, 'interactive_network_main_{}.html')  # GNN交互式可视化模板
        self.gnn_analysis_results = os.path.join(self.gnn_results_dir, 'gnn_analysis_results.json')  # GNN分析结果文件
        
        # GCN相关路径
        self.gcn_training_plot = os.path.join(self.gnn_results_dir, 'gcn_training.png')  # GCN训练曲线
        self.gcn_topology_png = os.path.join(self.gnn_results_dir, 'gcn_topology.png')  # GCN拓扑静态可视化
        self.gcn_interactive_topology = os.path.join(self.gnn_results_dir, 'gcn_interactive_topology.html')  # GCN拓扑交互式可视化
        self.gcn_topology_data = os.path.join(self.gnn_results_dir, 'gcn_topology_data.json')  # GCN拓扑数据
        self.gcn_real_pos_topology_png = os.path.join(self.gnn_results_dir, 'gcn_topology_real_positions.png')  # 基于真实位置的GCN拓扑静态可视化
        
        # GAT相关路径
        self.gat_training_plot = os.path.join(self.gnn_results_dir, 'gat_training.png')  # GAT训练曲线
        self.gat_topology_png = os.path.join(self.gnn_results_dir, 'gat_topology.png')  # GAT拓扑静态可视化
        self.gat_interactive_topology = os.path.join(self.gnn_results_dir, 'gat_interactive_topology.html')  # GAT拓扑交互式可视化
        self.gat_topology_data = os.path.join(self.gnn_results_dir, 'gat_topology_data.json')  # GAT拓扑数据
        
        # 基本交互式可视化路径
        self.interactive_neuron_network = os.path.join(self.interactive_dir, 'interactive_neuron_network.html')  # 基本交互式神经元网络
        
        # 拓扑分析所需数据文件路径
        self.position_data_file = os.path.join(self.data_dir, f'{self.data_identifier}_Max_position.csv')  # 神经元位置数据
        self.background_image = os.path.join(self.data_dir, f'{self.data_identifier}_Max.png')  # 背景图像
        
        # 模型文件路径
        self.model_path = os.path.join(self.model_dir, f'neuron_lstm_model_{self.data_identifier}.pth')  # 训练好的模型文件路径
        
        # 训练结果文件路径配置
        self.loss_plot = os.path.join(self.train_dir, f'training_loss_{self.data_identifier}.png')  # 训练损失曲线图
        self.accuracy_plot = os.path.join(self.train_dir, f'accuracy_curves_{self.data_identifier}.png')  # 准确率曲线图
        self.cluster_plot = os.path.join(self.train_dir, f'cluster_visualization_{self.data_identifier}.png')  # 聚类可视化图
        self.metrics_log = os.path.join(self.train_dir, f'training_metrics_{self.data_identifier}.csv')  # 训练指标日志
        self.error_log = os.path.join(self.train_dir, f'error_log_{self.data_identifier}.txt')  # 错误日志文件
        
        # 分析结果文件路径配置
        self.temporal_pattern_dir = os.path.join(self.analysis_dir, f'temporal_patterns_{self.data_identifier}')  # 时间模式分析目录
        self.correlation_plot = os.path.join(self.analysis_dir, f'behavior_neuron_correlation_{self.data_identifier}.png')  # 行为-神经元相关性图
        self.transition_plot = os.path.join(self.analysis_dir, f'behavior_transitions_{self.data_identifier}.png')  # 行为转换概率图
        self.key_neurons_plot = os.path.join(self.analysis_dir, f'key_neurons_{self.data_identifier}.png')  # 关键神经元分析图
        self.network_plot = os.path.join(self.analysis_dir, f'behavior_neuron_network_{self.data_identifier}.png')  # 行为-神经元网络图
        
        # 结果数据文件路径配置
        self.temporal_correlation_dir = os.path.join(self.analysis_dir, f'temporal_correlations_{self.data_identifier}')  # 时间相关性分析目录
        self.behavior_importance_csv = os.path.join(self.analysis_dir, f'behavior_importance_{self.data_identifier}.csv')  # 行为重要性数据
        self.neuron_specificity_json = os.path.join(self.analysis_dir, f'neuron_specificity_{self.data_identifier}.json')  # 神经元特异性数据
        self.statistical_results_csv = os.path.join(self.analysis_dir, f'statistical_analysis_{self.data_identifier}.csv')  # 统计分析结果
        
        # 神经网络分析和效应大小数据文件
        self.network_analysis_file = os.path.join(self.analysis_dir, f'network_analysis_results.json')  # 网络分析结果
        self.neuron_effect_file = os.path.join(self.analysis_dir, f'neuron_effect_sizes.csv')  # 神经元效应大小
        
        # LSTM模型超参数配置
        self.sequence_length: int = 10     # 序列长度：用于LSTM的输入序列长度
        self.hidden_size: int = 256        # 隐藏层大小：LSTM隐藏状态的维度
        self.num_layers: int = 3           # LSTM层数：模型中LSTM层的数量
        self.batch_size: int = 64          # 批次大小：训练时的批量大小
        self.learning_rate: float = 0.001  # 学习率：模型训练的学习率
        self.num_epochs: int = 100         # 训练轮数：模型训练的总轮数
        self.n_clusters: int = 6           # 聚类数量：K-means聚类的类别数
        self.test_size: float = 0.2        # 测试集比例：数据集中测试集的占比
        self.random_seed: int = 42         # 随机种子：确保结果可重复性
        
        self.weight_decay: float = 1e-4    # 权重衰减：优化器的权重衰减参数
        self.early_stopping: bool = False  # 早停：是否启用早停机制
         
        # 神经网络高级参数 - 新添加
        self.latent_dim: int = 32          # 潜在维度：自编码器的潜在特征维度
        self.num_heads: int = 4            # 注意力头数：多头注意力机制的头数量
        self.dropout: float = 0.2          # Dropout率：防止过拟合的神经元随机失活比例
        
        # 行为标签配置
        self.include_cd1_behavior: bool = True  # 是否在分析中纳入CD1行为标签
        
        # 分析参数配置
        self.analysis_params: Dict[str, Any] = {
            # 基本参数
            'min_samples_per_behavior': 10,  # 每种行为的最小样本数要求
            'correlation_windows': [10, 20, 50],  # 时间相关性分析的窗口大小列表
            'behavior_merge_threshold': 0.8,  # 行为合并的相似度阈值
            'neuron_significance_threshold': 1.0,  # 神经元显著性的阈值
            'temporal_window_size': 50,  # 时间窗口分析的大小
            'top_neurons_count': 5,  # 每个行为选择的关键神经元数量
            'p_value_threshold': 0.05,  # 统计检验的显著性水平
            'effect_size_threshold': 0.8,  # 效应量的显著性阈值
            'gradient_clip_norm': 1.0,  # 梯度裁剪的最大范数
            'weight_decay': 0.01,  # AdamW优化器的权重衰减系数
            
            # 新增数据集划分参数
            'use_time_aware_split': False,  # 是否使用时间感知的数据集划分（适合时间序列）
            'train_ratio': 0.6,  # 训练集比例
            'val_ratio': 0.2,   # 验证集比例
            'test_ratio': 0.2,  # 测试集比例
            
            # 类别不平衡处理参数
            'handle_class_imbalance': True,  # 是否处理类别不平衡
            'use_weighted_loss': True,  # 是否使用加权损失函数
            'use_sampling_techniques': False,  # 是否使用重采样技术
            
            # 早停机制参数
            'early_stopping_enabled': False,  # 是否启用早停机制
            'early_stopping_patience': 20,   # 早停耐心值
            
            # 新增的增强型LSTM模型参数
            'latent_dim': 32,  # 自编码器潜在空间维度
            'num_heads': 4,    # 多头注意力的头数
            'dropout': 0.2,    # Dropout率
            'reconstruction_loss_weight': 0.1,  # 重构损失的权重
            'attention_dropout': 0.1,  # 注意力机制的dropout率
            'autoencoder_hidden_dim': 128,  # 自编码器隐藏层维度
            
            # 新增的网络拓扑分析参数
            'correlation_threshold': 0.2,  # 构建功能连接网络的相关性阈值
            'min_module_size': 3,  # 功能模块的最小神经元数量
            'max_modules': 10,  # 最大功能模块数量
            'edge_weight_threshold': 0.4,  # 边权重阈值
            'community_resolution': 1.0,  # 社区检测的分辨率参数
            
            # GNN参数配置
            'gnn_epochs': 300,  # 恢复更多训练轮数
            'gnn_learning_rate': 0.008,  # 提高初始学习率
            'gnn_weight_decay': 1e-3,  # 减弱权重衰减强度
            'gnn_dropout': 0.3,  # 减弱Dropout强度
            'gnn_early_stop_patience': 20,  # 增加早停耐心值，允许更充分训练
            
            # GNN相关目录
            'gnn_results_dir': 'gnn_results',  # GNN结果子目录
            
            # 时间序列GNN参数
            'temporal_window_size': 10,  # 时间窗口大小
            'temporal_stride': 5,  # 时间窗口滑动步长
            
            # GCN增强模型参数
            'gcn_hidden_channels': 128,  # 增加GCN隐藏层维度
            'gcn_num_layers': 4,  # GCN层数
            'gcn_heads': 4,  # GCN注意力头数
            'gcn_use_batch_norm': True,  # 是否使用批归一化
            'gcn_activation': 'leaky_relu',  # 激活函数类型
            'gcn_alpha': 0.2,  # LeakyReLU的alpha参数
            'gcn_residual': True,  # 是否使用残差连接
            
            # GAT模型参数
            'gat_heads': 4,  # GAT注意力头数
            'gat_hidden_channels': 128,  # 增加GAT隐藏层维度
            'gat_dropout': 0.3,  # 减弱GAT特定的Dropout率
            'gat_residual': True,  # 启用残差连接
            'gat_num_layers': 3,  # GAT层数
            'gat_alpha': 0.2,  # LeakyReLU的alpha参数
            'gat_jk_mode': 'max',  # 跳跃连接模式：max, lstm, cat
            'gat_early_stopping_enabled': False,  # 控制GAT模型是否使用早停
            'gat_patience': 20,  # GAT早停机制的耐心值
            
        }
        
        # 可视化参数配置
        self.visualization_params: Dict[str, Any] = {
            'figure_sizes': {  # 不同类型图表的尺寸设置
                'correlation': (15, 10),  # 相关性图尺寸
                'temporal': (15, 5),      # 时间序列图尺寸
                'transitions': (10, 8),    # 转换概率图尺寸
                'key_neurons': (15, 8),    # 关键神经元图尺寸
                'network': (20, 15),       # 网络图尺寸
                'metrics': (15, 5),        # 训练指标图尺寸
                'attention': (20, 10),     # 注意力权重图尺寸
                'autoencoder': (15, 5)     # 自编码器重构图尺寸
            },
            'colormaps': {  # 不同类型图表的颜色方案
                'correlation': 'coolwarm',  # 相关性图的颜色方案
                'transitions': 'YlOrRd',    # 转换图的颜色方案
                'network': 'viridis',       # 网络图的颜色方案
                'clusters': 'viridis',      # 聚类图的颜色方案
                'attention': 'RdYlBu_r'     # 注意力权重图的颜色方案
            },
            'dpi': 300,              # 图像分辨率
            'save_format': 'png',    # 图像保存格式
            'font_size': 10,         # 字体大小
            'line_width': 2,         # 线条宽度
            'attention_plot_style': {
                'cmap': 'RdYlBu_r',
                'interpolation': 'nearest',
                'aspect': 'auto'
            },
            
            # 拓扑分析可视化参数
            'topology': {
                'use_background': True,           # 是否使用背景图
                'node_size': 10,                  # 节点大小（默认15，修改为10）
                'node_text_position': 'middle center',  # 节点文本位置
                'edge_width': 1,                  # 边的宽度（默认2，修改为1）
                'edge_color': 'black',            # 边的颜色
                'background_opacity': 0.8,        # 背景图透明度
                'frame_duration': 200,            # 帧持续时间（毫秒）
                'color_scheme': 'tab20',          # 颜色方案
                'max_groups': 20,                 # 最大组数
                'gif_fps': 5,                     # GIF帧率
                'edge_weight_threshold': 0.4      # 边权重阈值，只显示权重大于此值的边
            }
        }
        
        # GNN 分析配置
        self.use_gnn: bool = True  # 是否使用GNN分析
        
        # GAT模型控制开关
        self.use_gat: bool = False  # 是否使用GAT模型
        
        # 设置目录
        self.setup_directories()
        
        # 验证路径
        self.validate_paths()
        
    def setup_directories(self) -> None:
        """
        创建必要的目录结构并验证权限
        
        确保分析和结果所需的所有目录都存在，并验证它们具有适当的写入权限。
        如果目录不存在，将自动创建；如果无法创建或写入，则抛出异常。
        
        异常
        ----------
        RuntimeError
            当无法创建必要的目录结构时抛出
        PermissionError
            当无法在目录中写入文件时抛出
        """
        directories = [
            self.output_dir,           # 结果输出目录
            self.model_dir,            # 模型保存目录
            self.analysis_dir,         # 分析结果目录
            self.train_dir,            # 训练结果目录
            self.temporal_pattern_dir, # 时间模式分析目录
            self.temporal_correlation_dir,  # 时间相关性分析目录
            self.topology_dir,         # 拓扑分析目录
            self.gnn_results_dir,      # GNN结果目录
            self.interactive_dir       # 交互式可视化目录
        ]
        
        created_dirs = []
        try:
            for directory in directories:
                if not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)
                    created_dirs.append(directory)
                    print(f"创建目录: {directory}")
                
            # 验证目录是否可写
            test_file_path = os.path.join(self.train_dir, 'test_write.tmp')
            try:
                with open(test_file_path, 'w') as f:
                    f.write('test')
                os.remove(test_file_path)
            except Exception as e:
                raise PermissionError(f"无法在目录中写入文件: {str(e)}")
                
        except Exception as e:
            # 如果创建过程中出错，尝试删除已创建的目录
            for dir_path in created_dirs:
                try:
                    if os.path.exists(dir_path):
                        os.rmdir(dir_path)
                except:
                    pass
            raise RuntimeError(f"创建目录结构失败: {str(e)}")
        
    def validate_paths(self) -> None:
        """
        验证必要文件的存在性
        
        检查关键数据文件和模型文件是否存在，以及目录结构是否正确。
        如果缺少必要文件或目录，或者没有适当的访问权限，则抛出异常。
        
        异常
        ----------
        FileNotFoundError
            当必要的数据文件未找到时抛出
        NotADirectoryError
            当必要的目录不存在时抛出
        PermissionError
            当没有目录的写入权限时抛出
        """
        # 检查数据文件
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"数据文件未找到: {self.data_file}")
        
        # 检查目录结构
        required_dirs = {
            '输出目录': self.output_dir,
            '模型目录': self.model_dir,
            '分析目录': self.analysis_dir,
            '训练目录': self.train_dir
        }
        
        for name, path in required_dirs.items():
            if not os.path.exists(path):
                raise NotADirectoryError(f"{name}不存在: {path}")
            if not os.access(path, os.W_OK):
                raise PermissionError(f"没有{name}的写入权限: {path}")
            
    def get_temporal_pattern_path(self, behavior: str) -> str:
        """
        获取特定行为的时间模式图保存路径
        
        根据给定的行为类型名称，生成对应的时间模式分析图文件路径。
        
        参数
        ----------
        behavior : str
            行为类型名称
            
        返回
        ----------
        str
            对应行为的时间模式图文件完整路径
        """
        return os.path.join(self.temporal_pattern_dir, f'temporal_pattern_{behavior}.png')
    
    def get_temporal_correlation_path(self, window_size: int) -> str:
        """
        获取特定窗口大小的时间相关性图保存路径
        
        根据给定的时间窗口大小，生成对应的时间相关性分析图文件路径。
        
        参数
        ----------
        window_size : int
            时间窗口大小
            
        返回
        ----------
        str
            对应窗口大小的时间相关性图文件完整路径
        """
        return os.path.join(self.temporal_correlation_dir, f'temporal_correlation_{window_size}.png') 