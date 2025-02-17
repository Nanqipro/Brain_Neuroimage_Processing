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
        self.data_file = os.path.join(self.data_dir, 'Day6_with_behavior_labels_filled.xlsx')  # 原始数据文件
        
        # 输出目录配置
        self.output_dir = os.path.join(self.base_dir, 'results')  # 结果输出总目录
        self.model_dir = os.path.join(self.base_dir, 'models')    # 模型保存目录
        self.analysis_dir = os.path.join(self.output_dir, 'analysis')  # 分析结果目录
        
        # 模型文件路径
        self.model_path = os.path.join(self.model_dir, 'neuron_lstm_model.pth')  # 训练好的模型文件路径
        
        # 分析结果文件路径配置
        self.correlation_plot = os.path.join(self.analysis_dir, 'behavior_neuron_correlation.png')  # 行为-神经元相关性图
        self.transition_plot = os.path.join(self.analysis_dir, 'behavior_transitions.png')  # 行为转换概率图
        self.key_neurons_plot = os.path.join(self.analysis_dir, 'key_neurons.png')  # 关键神经元分析图
        self.temporal_pattern_dir = os.path.join(self.analysis_dir, 'temporal_patterns')  # 时间模式分析目录
        self.network_plot = os.path.join(self.analysis_dir, 'behavior_neuron_network.png')  # 行为-神经元网络图
        
        # 结果数据文件路径配置
        self.behavior_importance_csv = os.path.join(self.analysis_dir, 'behavior_importance.csv')  # 行为重要性数据
        self.neuron_specificity_json = os.path.join(self.analysis_dir, 'neuron_specificity.json')  # 神经元特异性数据
        self.statistical_results_csv = os.path.join(self.analysis_dir, 'statistical_analysis.csv')  # 统计分析结果
        self.temporal_correlation_dir = os.path.join(self.analysis_dir, 'temporal_correlations')  # 时间相关性分析目录
        
        # 模型超参数配置
        self.sequence_length = 10    # 序列长度：用于LSTM的输入序列长度
        self.hidden_size = 128       # 隐藏层大小：LSTM隐藏状态的维度
        self.num_layers = 2          # LSTM层数：模型中LSTM层的数量
        self.batch_size = 32         # 批次大小：训练时的批量大小
        self.learning_rate = 0.001   # 学习率：模型训练的学习率
        self.num_epochs = 50         # 训练轮数：模型训练的总轮数
        self.n_clusters = 5          # 聚类数量：K-means聚类的类别数
        self.test_size = 0.2         # 测试集比例：数据集中测试集的占比
        self.random_seed = 42        # 随机种子：确保结果可重复性
        
        # 分析参数配置
        self.analysis_params = {
            'min_samples_per_behavior': 10,  # 每种行为的最小样本数要求
            'correlation_windows': [10, 20, 50, 100],  # 时间相关性分析的窗口大小列表
            'behavior_merge_threshold': 0.8,  # 行为合并的相似度阈值
            'neuron_significance_threshold': 1.0,  # 神经元显著性的阈值
            'temporal_window_size': 50,  # 时间窗口分析的大小
            'top_neurons_count': 5,  # 每个行为选择的关键神经元数量
            'p_value_threshold': 0.05,  # 统计检验的显著性水平
            'effect_size_threshold': 0.5  # 效应量的显著性阈值
        }
        
        # 可视化参数配置
        self.visualization_params = {
            'figure_sizes': {  # 不同类型图表的尺寸设置
                'correlation': (15, 10),  # 相关性图尺寸
                'temporal': (15, 5),      # 时间序列图尺寸
                'transitions': (10, 8),    # 转换概率图尺寸
                'key_neurons': (15, 8),    # 关键神经元图尺寸
                'network': (20, 15)        # 网络图尺寸
            },
            'colormaps': {  # 不同类型图表的颜色方案
                'correlation': 'coolwarm',  # 相关性图的颜色方案
                'transitions': 'YlOrRd',    # 转换图的颜色方案
                'network': 'viridis'        # 网络图的颜色方案
            },
            'dpi': 300,              # 图像分辨率
            'save_format': 'png',    # 图像保存格式
            'font_size': 10,         # 字体大小
            'line_width': 2          # 线条宽度
        }
        
    def setup_directories(self):
        """
        创建必要的目录结构
        确保所有需要的输出目录都存在
        """
        directories = [
            self.output_dir,           # 结果输出目录
            self.model_dir,            # 模型保存目录
            self.analysis_dir,         # 分析结果目录
            self.temporal_pattern_dir, # 时间模式分析目录
            self.temporal_correlation_dir  # 时间相关性分析目录
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
    def validate_paths(self):
        """
        验证必要文件的存在性
        检查数据文件和模型文件是否存在
        如果缺少必要文件，抛出FileNotFoundError异常
        """
        required_files = {
            'Data file': self.data_file,    # 数据文件
            'Model file': self.model_path    # 模型文件
        }
        
        missing_files = []
        for name, path in required_files.items():
            if not os.path.exists(path):
                missing_files.append(f"{name}: {path}")
        
        if missing_files:
            raise FileNotFoundError("Required files not found:\n" + "\n".join(missing_files))
            
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