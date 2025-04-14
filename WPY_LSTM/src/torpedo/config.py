import os
from typing import Optional

class LSTMConfig:
    """
    LSTM 分析配置类
    """
    def __init__(self) -> None:
        # --- 基础路径设置 ---
        # 假设脚本在 src/torpedo/ 下运行，获取项目根目录 (LSTM/)
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.base_dir = os.path.join(self.project_root, 'src', 'torpedo') # 当前工作目录
        
        # --- 数据文件路径 ---
        # !!! 重要: 请根据实际情况修改数据文件路径 !!!
        # self.data_file = os.path.join(self.project_root, 'datasets', 'EMtrace.xlsx') # 原始示例
        self.data_file = os.path.join(self.project_root, 'datasets', 'processed_EMtrace.xlsx') # 高架环境数据
        # self.data_file = '/path/to/your/specific/datafile.xlsx' # 或者使用绝对路径
        
        # 从数据文件名提取标识符 (用于结果命名)
        try:
            self.data_identifier = os.path.splitext(os.path.basename(self.data_file))[0]
        except:
            self.data_identifier = 'lstm_analysis' # 默认标识符
            
        # --- 输出目录 ---
        self.output_dir = os.path.join(self.project_root, 'results', 'torpedo_lstm', f'analysis_{self.data_identifier}') # 分析结果总目录
        self.model_dir = os.path.join(self.project_root, 'models', 'torpedo_lstm')    # 模型和 Scaler 保存目录
        self.log_dir = os.path.join(self.output_dir, 'logs')                          # 日志文件目录
        self.plot_dir = os.path.join(self.output_dir, 'plots')                        # 图像保存目录
        
        # --- 模型和 Scaler 文件路径 ---
        self.model_path = os.path.join(self.model_dir, f'lstm_model_{self.data_identifier}.pth')
        self.scaler_path = os.path.join(self.model_dir, f'scaler_{self.data_identifier}.joblib')
        
        # --- 日志文件路径 ---
        self.train_log = os.path.join(self.log_dir, f'training_log_{self.data_identifier}.log')
        self.eval_log = os.path.join(self.log_dir, f'evaluation_log_{self.data_identifier}.log')
        self.error_log = os.path.join(self.log_dir, f'error_log_{self.data_identifier}.txt')
        self.metrics_log = os.path.join(self.log_dir, f'training_metrics_{self.data_identifier}.csv') # 训练指标 csv
        
        # --- 结果文件路径 ---
        self.eval_results_json = os.path.join(self.output_dir, f'test_evaluation_results_{self.data_identifier}.json')
        
        # --- 绘图文件路径 ---
        self.accuracy_plot = os.path.join(self.plot_dir, f'accuracy_curves_{self.data_identifier}.png')
        self.loss_plot = os.path.join(self.plot_dir, f'loss_curves_{self.data_identifier}.png') # 单独的损失曲线图
        self.confusion_matrix_plot = os.path.join(self.plot_dir, f'test_confusion_matrix_{self.data_identifier}.png')
        
        # --- 数据分割参数 ---
        self.val_test_split_ratio: float = 0.3 # 验证集和测试集合并占总数据的比例 (例如 0.3 代表 30%)
        self.test_split_ratio: float = 0.5     # 测试集占(验证集+测试集)的比例 (例如 0.5 代表测试集和验证集各占15%)
        
        # --- LSTM 模型超参数 ---
        self.sequence_length: int = 30     # 输入序列长度 (增加)
        self.hidden_size: int = 128        # LSTM隐藏状态维度 (降低)
        self.num_layers: int = 1           # LSTM层数 (降低)
        self.latent_dim: int = 32          # (虽然模型不用 AE，但配置项保留)
        self.num_heads: int = 4            # (虽然模型不用 Attention，但配置项保留)
        self.dropout: float = 0.4          # Dropout率 (暂时保持)
        
        # --- 训练参数 ---
        self.batch_size: int = 64          # 批次大小
        self.learning_rate: float = 0.0001 # 学习率 (显著降低)
        self.num_epochs: int = 100         # 训练轮数
        self.weight_decay: float = 5e-3    # 优化器权重衰减 (增加)
        self.gradient_clip_norm: float = 1.0 # 梯度裁剪范数
        self.reconstruction_loss_weight: float = 0 # 重构损失权重 (设为 0，因为 AE 已移除)
        
        # --- 学习率调度器参数 ---
        self.lr_scheduler_patience: int = 10 # ReduceLROnPlateau 的耐心值
        
        # --- 早停参数 ---
        self.early_stopping_enabled: bool = True  # 是否启用早停
        self.early_stopping_patience: int = 20    # 早停耐心值 (恢复为 20)
        
        # --- 交叉验证参数 ---
        self.k_folds: int = 5               # K-Fold 交叉验证折数
        self.max_episode_len: Optional[int] = 100 # (新增) 基于片段 K-Fold 时，单个片段最大长度 (None 表示不分割)
        
        # --- 其他 ---
        self.random_seed: int = 42         # 随机种子
        self.include_cd1_behavior: bool = True # 是否包含 CD1 行为 (来自旧配置)
        
        # --- 自动创建目录 ---
        self._setup_directories()

    def _setup_directories(self) -> None:
        """创建必要的输出目录"""
        directories = [
            self.output_dir,
            self.model_dir,
            self.log_dir,
            self.plot_dir
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

# 可以添加一个函数来加载或验证配置
# def load_config():
#     return LSTMConfig()

if __name__ == '__main__':
    # 测试配置是否能正确初始化并创建目录
    try:
        config = LSTMConfig()
        print("LSTM 配置加载成功!")
        print(f"数据文件: {config.data_file}")
        print(f"模型文件: {config.model_path}")
        print(f"Scaler文件: {config.scaler_path}")
        print(f"输出目录: {config.output_dir}")
        print(f"模型目录: {config.model_dir}")
        print(f"日志目录: {config.log_dir}")
        print(f"绘图目录: {config.plot_dir}")
        # 检查目录是否创建
        if not os.path.exists(config.output_dir):
            print(f"错误: 输出目录 {config.output_dir} 未创建!")
        if not os.path.exists(config.model_dir):
            print(f"错误: 模型目录 {config.model_dir} 未创建!")
    except Exception as e:
        print(f"加载 LSTM 配置时出错: {e}") 