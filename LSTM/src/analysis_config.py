import os

class AnalysisConfig:
    """
    分析配置类：管理神经元行为分析项目的所有配置参数
    包括：
    1. 文件路径配置
    2. 模型参数配置
    3. 分析参数配置
    4. 可视化参数配置
    """
    def __init__(self):
        # 基础目录配置
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录
        
        # 数据路径配置
        self.data_dir = os.path.join(self.base_dir, 'datasets')  # 数据集目录
        self.data_file = os.path.join(self.data_dir, 'EMtrace.xlsx')  # 原始数据文件
        self.data_identifier = 'EMtrace'  # 从数据文件名提取标识符
        
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
        self.correlation_plot = os.path.join(self.analysis_dir, f'behavior_neuron_correlation_{self.data_identifier}.png')  # 行为-神经元相关性图
        self.transition_plot = os.path.join(self.analysis_dir, f'behavior_transitions_{self.data_identifier}.png')  # 行为转换概率图
        self.key_neurons_plot = os.path.join(self.analysis_dir, f'key_neurons_{self.data_identifier}.png')  # 关键神经元分析图
        self.temporal_pattern_dir = os.path.join(self.analysis_dir, f'temporal_patterns_{self.data_identifier}')  # 时间模式分析目录
        self.network_plot = os.path.join(self.analysis_dir, f'behavior_neuron_network_{self.data_identifier}.png')  # 行为-神经元网络图
        
        # 结果数据文件路径配置
        self.behavior_importance_csv = os.path.join(self.analysis_dir, f'behavior_importance_{self.data_identifier}.csv')  # 行为重要性数据
        self.neuron_specificity_json = os.path.join(self.analysis_dir, f'neuron_specificity_{self.data_identifier}.json')  # 神经元特异性数据
        self.statistical_results_csv = os.path.join(self.analysis_dir, f'statistical_analysis_{self.data_identifier}.csv')  # 统计分析结果
        self.temporal_correlation_dir = os.path.join(self.analysis_dir, f'temporal_correlations_{self.data_identifier}')  # 时间相关性分析目录
        
        # 神经网络分析和效应大小数据文件
        self.network_analysis_file = os.path.join(self.analysis_dir, f'network_analysis_results.json')  # 网络分析结果
        self.neuron_effect_file = os.path.join(self.analysis_dir, f'neuron_effect_sizes.csv')  # 神经元效应大小
        
        # 模型超参数配置
        self.sequence_length = 10     # 序列长度：用于LSTM的输入序列长度
        self.hidden_size = 256        # 隐藏层大小：LSTM隐藏状态的维度
        self.num_layers = 3           # LSTM层数：模型中LSTM层的数量
        self.batch_size = 64          # 批次大小：训练时的批量大小
        self.learning_rate = 0.001    # 学习率：模型训练的学习率
        self.num_epochs = 100         # 训练轮数：模型训练的总轮数
        self.n_clusters = 5           # 聚类数量：K-means聚类的类别数
        self.test_size = 0.2          # 测试集比例：数据集中测试集的占比
        self.random_seed = 42         # 随机种子：确保结果可重复性
        
        # 分析参数配置
        self.analysis_params = {
            'min_samples_per_behavior': 10,  # 每种行为的最小样本数要求
            'correlation_windows': [10, 20, 50, 100],  # 时间相关性分析的窗口大小列表
            'behavior_merge_threshold': 0.8,  # 行为合并的相似度阈值
            'neuron_significance_threshold': 1.0,  # 神经元显著性的阈值
            'temporal_window_size': 50,  # 时间窗口分析的大小
            'top_neurons_count': 5,  # 每个行为选择的关键神经元数量
            'p_value_threshold': 0.05,  # 统计检验的显著性水平
            'effect_size_threshold': 0.5,  # 效应量的显著性阈值
            'gradient_clip_norm': 1.0,  # 梯度裁剪的最大范数
            'weight_decay': 0.01,  # AdamW优化器的权重衰减系数
            
            # 新增的增强型LSTM模型参数
            'latent_dim': 32,  # 自编码器潜在空间维度
            'num_heads': 4,    # 多头注意力的头数
            'dropout': 0.2,    # Dropout率
            'reconstruction_loss_weight': 0.1,  # 重构损失的权重
            'attention_dropout': 0.1,  # 注意力机制的dropout率
            'autoencoder_hidden_dim': 128,  # 自编码器隐藏层维度
            
            # 新增的网络拓扑分析参数
            'correlation_threshold': 0.3,  # 构建功能连接网络的相关性阈值
            'min_module_size': 3,  # 功能模块的最小神经元数量
            'max_modules': 10,  # 最大功能模块数量
            'edge_weight_threshold': 0.5,  # 边权重阈值
            'community_resolution': 1.0,  # 社区检测的分辨率参数
        }
        
        # 可视化参数配置
        self.visualization_params = {
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
                'edge_weight_threshold': 0.3      # 边权重阈值，只显示权重大于此值的边
            }
        }
        
    def setup_directories(self):
        """
        创建必要的目录结构并验证权限
        确保所有需要的输出目录都存在
        """
        directories = [
            self.output_dir,           # 结果输出目录
            self.model_dir,            # 模型保存目录
            self.analysis_dir,         # 分析结果目录
            self.train_dir,            # 训练结果目录
            self.temporal_pattern_dir, # 时间模式分析目录
            self.temporal_correlation_dir,  # 时间相关性分析目录
            self.topology_dir          # 拓扑分析目录
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
        
    def validate_paths(self):
        """
        验证必要文件的存在性
        检查数据文件和模型文件是否存在
        如果缺少必要文件，抛出FileNotFoundError异常
        """
        # 首先确保目录结构存在
        self.setup_directories()
        
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
            
    def get_temporal_pattern_path(self, behavior):
        """
        获取特定行为的时间模式图保存路径
        参数：
            behavior: 行为类型名称
        返回：
            对应行为的时间模式图文件路径
        """
        return os.path.join(self.temporal_pattern_dir, f'temporal_pattern_{behavior}.png')
    
    def get_temporal_correlation_path(self, window_size):
        """
        获取特定窗口大小的时间相关性图保存路径
        参数：
            window_size: 时间窗口大小
        返回：
            对应窗口大小的时间相关性图文件路径
        """
        return os.path.join(self.temporal_correlation_dir, f'temporal_correlation_{window_size}.png') 