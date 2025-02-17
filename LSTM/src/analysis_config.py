import os

class AnalysisConfig:
    def __init__(self):
        # Base directory
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Data paths
        self.data_dir = os.path.join(self.base_dir, 'datasets')
        self.data_file = os.path.join(self.data_dir, 'Day6_with_behavior_labels_filled.xlsx')
        
        # Output directories
        self.output_dir = os.path.join(self.base_dir, 'results')
        self.model_dir = os.path.join(self.base_dir, 'models')
        self.analysis_dir = os.path.join(self.output_dir, 'analysis')
        
        # Model files
        self.model_path = os.path.join(self.model_dir, 'neuron_lstm_model.pth')
        
        # Analysis result files
        self.correlation_plot = os.path.join(self.analysis_dir, 'behavior_neuron_correlation.png')
        self.transition_plot = os.path.join(self.analysis_dir, 'behavior_transitions.png')
        self.key_neurons_plot = os.path.join(self.analysis_dir, 'key_neurons.png')
        self.temporal_pattern_dir = os.path.join(self.analysis_dir, 'temporal_patterns')
        self.network_plot = os.path.join(self.analysis_dir, 'behavior_neuron_network.png')
        
        # Results data files
        self.behavior_importance_csv = os.path.join(self.analysis_dir, 'behavior_importance.csv')
        self.neuron_specificity_json = os.path.join(self.analysis_dir, 'neuron_specificity.json')
        self.statistical_results_csv = os.path.join(self.analysis_dir, 'statistical_analysis.csv')
        self.temporal_correlation_dir = os.path.join(self.analysis_dir, 'temporal_correlations')
        
        # Model parameters
        self.sequence_length = 10
        self.hidden_size = 128
        self.num_layers = 2
        self.batch_size = 32
        self.learning_rate = 0.001
        self.num_epochs = 50
        self.n_clusters = 5
        self.test_size = 0.2
        self.random_seed = 42
        
        # Analysis parameters
        self.analysis_params = {
            'min_samples_per_behavior': 10,  # 每种行为的最小样本数
            'correlation_windows': [10, 20, 50, 100],  # 时间相关性分析窗口
            'behavior_merge_threshold': 0.8,  # 行为合并阈值
            'neuron_significance_threshold': 1.0,  # 神经元显著性阈值
            'temporal_window_size': 50,  # 时间窗口大小
            'top_neurons_count': 5,  # 每个行为的top神经元数量
            'p_value_threshold': 0.05,  # 统计显著性阈值
            'effect_size_threshold': 0.5  # 效应量阈值
        }
        
        # Visualization parameters
        self.visualization_params = {
            'figure_sizes': {
                'correlation': (15, 10),
                'temporal': (15, 5),
                'transitions': (10, 8),
                'key_neurons': (15, 8),
                'network': (20, 15)
            },
            'colormaps': {
                'correlation': 'coolwarm',
                'transitions': 'YlOrRd',
                'network': 'viridis'
            },
            'dpi': 300,
            'save_format': 'png',
            'font_size': 10,
            'line_width': 2
        }
        
    def setup_directories(self):
        """Create all necessary directories"""
        directories = [
            self.output_dir,
            self.model_dir,
            self.analysis_dir,
            self.temporal_pattern_dir,
            self.temporal_correlation_dir
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
    def validate_paths(self):
        """Validate required files and directories"""
        required_files = {
            'Data file': self.data_file,
            'Model file': self.model_path
        }
        
        missing_files = []
        for name, path in required_files.items():
            if not os.path.exists(path):
                missing_files.append(f"{name}: {path}")
        
        if missing_files:
            raise FileNotFoundError("Required files not found:\n" + "\n".join(missing_files))
            
    def get_temporal_pattern_path(self, behavior):
        """Get path for temporal pattern plot of specific behavior"""
        return os.path.join(self.temporal_pattern_dir, f'temporal_pattern_{behavior}.png')
    
    def get_temporal_correlation_path(self, window_size):
        """Get path for temporal correlation plot"""
        return os.path.join(self.temporal_correlation_dir, f'temporal_correlation_{window_size}.png') 