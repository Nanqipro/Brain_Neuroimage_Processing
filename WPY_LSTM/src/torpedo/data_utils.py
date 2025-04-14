import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans # 保留 KMeans 以便 apply_kmeans 正常工作
import re
import os
import warnings # 添加导入
from typing import Tuple, List, Dict, Any, Optional
from joblib import dump, load
import math

# 从同级目录导入配置 (使用相对导入)
from .config import LSTMConfig 

warnings.filterwarnings('ignore')

# --- Helper function to find and potentially split behavior episodes ---
def find_behavior_episodes(y: np.ndarray, max_len: Optional[int]) -> List[Tuple[int, int, int]]:
    """
    Identifies continuous episodes of the same behavior label and splits
    episodes longer than max_len into sub-episodes.

    Args:
        y: NumPy array of behavior labels.
        max_len: Maximum length for an episode. If None, episodes are not split.

    Returns:
        A list of tuples, where each tuple is (start_index, end_index, label).
        end_index is inclusive.
    """
    if len(y) == 0:
        return []

    episodes = []
    start_index = 0
    current_label = y[0]

    for i in range(1, len(y)):
        if y[i] != current_label:
            # End of the previous episode
            episode_start = start_index
            episode_end = i - 1
            episode_label = current_label
            episode_len = episode_end - episode_start + 1

            if max_len is not None and episode_len > max_len:
                # Split long episode
                num_sub_episodes = math.ceil(episode_len / max_len)
                for j in range(num_sub_episodes):
                    sub_start = episode_start + j * max_len
                    sub_end = min(sub_start + max_len - 1, episode_end)
                    if sub_start <= sub_end: # Ensure valid sub-episode
                         episodes.append((sub_start, sub_end, episode_label))
            else:
                # Add the original (short enough) episode
                episodes.append((episode_start, episode_end, episode_label))

            # Start of a new episode
            start_index = i
            current_label = y[i]

    # Add the last episode (and potentially split it)
    episode_start = start_index
    episode_end = len(y) - 1
    episode_label = current_label
    episode_len = episode_end - episode_start + 1

    if max_len is not None and episode_len > max_len:
        num_sub_episodes = math.ceil(episode_len / max_len)
        for j in range(num_sub_episodes):
            sub_start = episode_start + j * max_len
            sub_end = min(sub_start + max_len - 1, episode_end)
            if sub_start <= sub_end:
                episodes.append((sub_start, sub_end, episode_label))
    else:
         if episode_start <= episode_end: # Handle empty y case edge
            episodes.append((episode_start, episode_end, episode_label))

    return episodes

# --- Helper function to get data indices from episodes ---
def get_indices_from_episodes(episode_list: List[Tuple[int, int, int]]) -> np.ndarray:
    """Converts a list of episodes to a flat array of data indices."""
    indices = []
    for start, end, _ in episode_list:
        indices.extend(range(start, end + 1))
    return np.array(indices, dtype=int)

# --- 数据集类 ---
class NeuronDataset(Dataset):
    """
    神经元数据集类，用于处理序列数据
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, sequence_length: int):
        """
        初始化数据集。

        参数:
            X (np.ndarray): 标准化后的神经元活动数据 (样本数, 特征数)
            y (np.ndarray): 编码后的行为标签 (样本数,)
            sequence_length (int): 序列长度
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"数据 X 和标签 y 的样本数不匹配: {X.shape[0]} != {y.shape[0]}")
        if X.ndim != 2:
             raise ValueError(f"输入数据 X 应为 2 维 (样本数, 特征数)，实际为 {X.ndim} 维")
             
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y) # 确保是 LongTensor 用于损失计算
        self.sequence_length = sequence_length
        self._num_samples = len(self.X) - self.sequence_length + 1
        if self._num_samples <= 0:
             raise ValueError(f"数据长度 ({len(self.X)}) 过短，无法创建长度为 {self.sequence_length} 的序列。")

    def __len__(self) -> int:
        """返回数据集中序列的数量"""
        return self._num_samples
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取特定索引的样本 (序列及其对应的最后一个时间点的标签)"""
        if not (0 <= idx < self._num_samples):
             raise IndexError(f"索引 {idx} 超出范围 [0, {self._num_samples - 1}]")
             
        # 输入序列 X[idx] 到 X[idx + sequence_length - 1]
        sequence = self.X[idx : idx + self.sequence_length]
        # 目标标签是序列最后一个时间点对应的标签 y[idx + sequence_length - 1]
        target_label = self.y[idx + self.sequence_length - 1]
        
        return sequence, target_label

# --- 数据处理器类 ---
class NeuronDataProcessor:
    """
    神经元数据处理器类。
    (与之前 kmeans_lstm_analysis.py 中修改后的版本相同)
    """
    def __init__(self, config: LSTMConfig) -> None:
        self.config = config
        try:
            self.data = pd.read_excel(config.data_file)
            print(f"成功加载数据文件: {config.data_file}")
            print(f"数据形状: {self.data.shape}")
        except FileNotFoundError:
             print(f"错误: 数据文件未找到 {config.data_file}")
             raise
        except Exception as e:
            print(f"加载 Excel 文件时出错: {str(e)}")
            # 可以尝试其他读取方式，或直接抛出错误
            # 尝试读取 CSV
            try:
                 csv_path = config.data_file.replace('.xlsx', '.csv')
                 print(f"尝试读取 CSV 文件: {csv_path}")
                 self.data = pd.read_csv(csv_path)
                 print(f"成功加载 CSV 数据文件: {csv_path}")
                 print(f"数据形状: {self.data.shape}")
            except Exception as csv_e:
                 print(f"读取 CSV 文件也失败: {csv_e}")
                 raise e # 重新抛出原始 Excel 读取错误

        self.scaler = StandardScaler()
        self.scaler_fitted = False
        self.label_encoder = LabelEncoder()
        self.available_neuron_cols: List[str] = []

    def preprocess_data(self) -> Tuple[np.ndarray, np.ndarray]:
        # (代码与之前相同，返回未缩放的 X, y)
        if self.data.empty: raise ValueError("数据为空")
        neuron_pattern = re.compile(r'^n\d+$')
        neuron_cols = [col for col in self.data.columns if neuron_pattern.match(col)]
        # ... (识别和检查神经元列的代码)
        if not neuron_cols: raise ValueError("无法识别神经元数据列")
        neuron_cols.sort()
        self.available_neuron_cols = [col for col in neuron_cols if col in self.data.columns]
        print(f"可用神经元数量: {len(self.available_neuron_cols)} / {len(neuron_cols)}")
        try: X = self.data[self.available_neuron_cols].values.astype(float)
        except Exception as e: raise ValueError(f"提取神经元数据时出错: {e}")
        if np.isnan(X).any():
            print("发现缺失值，使用列均值填充")
            col_means = np.nanmean(X, axis=0)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(col_means, inds[1])
        if not np.isfinite(X).all():
            print("发现无限值，将替换为有限值")
            X = np.nan_to_num(X, nan=0.0, posinf=np.finfo(X.dtype).max, neginf=np.finfo(X.dtype).min)

        # --- 核心步骤：行为标签处理、映射和过滤 ---
        behavior_col = None
        for col_name in ['behavior', 'Behavior', 'label', 'Label', 'class', 'Class']:
            if col_name in self.data.columns:
                behavior_col = col_name
                break
        if behavior_col is None:
            raise ValueError("找不到行为标签列")

        # 原始行为标签
        original_behaviors = self.data[behavior_col].astype(str)
        
        # --- 定义核心类别映射 ---
        # 将相似行为合并到 'Closed', 'Middle', 'Open' 三大类
        label_map = {
            # Closed Arm Related
            'Closed-arm': 'Closed',
            'Closed-Armed-Exp': 'Closed',
            'Closed-arm-stiff': 'Closed',
            # Middle Zone Related
            'Middle-Zone': 'Middle',
            'Middle-Zone-stiff': 'Middle',
            # Open Arm Related
            'Open-arm': 'Open',
            'Open-Armed-Exp': 'Open',
            'Open-arm-probe': 'Open',
            # --- 以下类别将被忽略或移除 ---
            # 'Middle-Zone-to-Close-arm': 'Transition', # 或 'Middle', 'Closed' 或移除
            # 'Middle-Zone-to-Open-arm': 'Transition', # 或 'Middle', 'Open' 或移除
            # 'stiff': 'Other', # 或移除
            # 'Move': 'Other', # 或移除
            # 'Open-arm-exp-middle': 'Other', # 或移除
            # 'middle-exp-close-arm': 'Other', # 或移除
            # 'middle-exp-open-arm': 'Other', # 或移除
        }
        target_labels = ['Closed', 'Middle', 'Open'] # 我们期望的最终类别
        print(f"\n将行为标签映射到核心类别: {target_labels}")

        # 应用映射，不在映射中的标签设为 None 或特定标记 (如 'Ignore')
        mapped_behaviors = original_behaviors.map(label_map) # 不在 map 中的会自动变成 NaN
        
        # --- 过滤数据 ---
        # 保留映射到核心类别的行
        mask = mapped_behaviors.isin(target_labels)
        original_count = len(self.data)
        
        self.data = self.data[mask].reset_index(drop=True)
        X = X[mask]
        final_behaviors = mapped_behaviors[mask]
        
        filtered_count = len(self.data)
        print(f"根据核心类别过滤数据: 保留 {filtered_count} / {original_count} 条记录。")
        
        if filtered_count == 0:
            raise ValueError("过滤后没有剩余数据，请检查标签映射和原始数据。")
        # -------------------------------------
        
        # 对过滤和映射后的标签进行编码
        try:
            # 确保 LabelEncoder 只学习目标类别
            self.label_encoder.fit(target_labels) 
            y = self.label_encoder.transform(final_behaviors)
        except Exception as e:
            # ... (可以保留之前的备用手动编码逻辑，但基于 target_labels)
            print(f"编码过滤后的行为标签时出错: {e}")
            raise
        
        # 打印新的标签编码信息和计数
        print("\nFiltered Behavior label mapping and counts:")
        unique_labels_filtered, counts_filtered = np.unique(y, return_counts=True)
        for label_code in unique_labels_filtered:
            label_name = self.label_encoder.inverse_transform([label_code])[0]
            count = counts_filtered[label_code]
            print(f"- {label_name} (Code: {label_code}): {count} samples")
            
        print("\n数据过滤、映射和基础预处理完成。")
        return X, y

    def fit_scaler(self, X_train: np.ndarray) -> None:
        # (代码与之前相同)
        print("--- 在训练数据上拟合 StandardScaler ---")
        if X_train.ndim != 2: raise ValueError(f"拟合 scaler 的输入维度应为 2，实际为 {X_train.ndim}")
        if X_train.shape[0] == 0: raise ValueError("拟合 scaler 的训练数据不能为空。")
        try:
            self.scaler.fit(X_train); self.scaler_fitted = True
            print(f"StandardScaler 拟合完成。")
        except Exception as e: print(f"拟合 StandardScaler 时出错: {e}"); self.scaler_fitted = False; raise

    def scale_data(self, X: np.ndarray) -> np.ndarray:
        # (代码与之前相同)
        if not self.scaler_fitted: raise RuntimeError("StandardScaler 尚未拟合，请先调用 fit_scaler。")
        if X.ndim != 2: raise ValueError(f"进行 scale 的输入数据维度应为 2，实际为 {X.ndim}")
        print(f"--- 使用已拟合的 StandardScaler 转换数据 (形状: {X.shape}) ---")
        try: return self.scaler.transform(X)
        except Exception as e: print(f"转换数据时出错: {e}"); raise
        
    def apply_kmeans(self, X: np.ndarray, use_scaled: bool = True) -> Tuple[KMeans, np.ndarray]:
        # (代码与之前相同)
        if use_scaled:
             print("KMeans将在标准化数据上执行。")
             if not self.scaler_fitted: 
                 print("警告：Scaler尚未拟合，临时拟合用于KMeans。"); temp_scaler = StandardScaler().fit(X); X_to_cluster = temp_scaler.transform(X)
             else: X_to_cluster = self.scale_data(X)
        else: print("KMeans将在未标准化数据上执行。"); X_to_cluster = X
        print(f"\n使用 {self.config.n_clusters} 个聚类进行K-means聚类") # config 中需要 n_clusters 参数
        kmeans = KMeans(n_clusters=self.config.n_clusters, random_state=self.config.random_seed, n_init=10)
        cluster_labels = kmeans.fit_predict(X_to_cluster)
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        print("\n聚类分布:"); [print(f"聚类 {l}: {c} 个样本") for l, c in zip(unique_labels, counts)]
        return kmeans, cluster_labels

# --- 数据分割函数 (按时间顺序) ---
def split_data(X: np.ndarray, y: np.ndarray, config: LSTMConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    将原始数据按时间顺序划分为训练集、验证集和测试集。
    直接返回 NumPy 数组。

    参数:
        X (np.ndarray): 未缩放的特征数据 (样本数, 特征数)。
        y (np.ndarray): 对应的标签数组 (样本数,)。
        config (LSTMConfig): 配置对象。

    返回:
        Tuple[np.ndarray, np.ndarray, ...]: (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    num_samples = X.shape[0]
    if num_samples != y.shape[0]:
        raise ValueError("分割时 X 和 y 的样本数不匹配。")

    val_test_ratio = config.val_test_split_ratio
    test_ratio_in_val_test = config.test_split_ratio

    if not (0 < val_test_ratio < 1) or not (0 < test_ratio_in_val_test < 1):
        raise ValueError("分割比例参数无效。")

    # 计算分割点索引
    num_val_test = int(num_samples * val_test_ratio)
    num_test = int(num_val_test * test_ratio_in_val_test)
    num_val = num_val_test - num_test
    num_train = num_samples - num_val_test

    if num_train <= 0 or num_val <= 0 or num_test <= 0:
        raise ValueError(f"数据集太小 ({num_samples})，无法按指定比例分割。")

    # 按时间顺序分割
    X_train = X[:num_train]
    y_train = y[:num_train]
    X_val = X[num_train : num_train + num_val]
    y_val = y[num_train : num_train + num_val]
    X_test = X[num_train + num_val :]
    y_test = y[num_train + num_val :]

    print(f"按时间顺序分割完成: 训练集 {X_train.shape}, 验证集 {X_val.shape}, 测试集 {X_test.shape}")
    
    # 可选：打印类别分布
    for name, y_subset in zip(["训练集", "验证集", "测试集"], [y_train, y_val, y_test]):
        unique, counts = np.unique(y_subset, return_counts=True)
        print(f"{name} 类别分布 (仅供参考): {dict(zip(unique, counts))}")

    return X_train, y_train, X_val, y_val, X_test, y_test

# --- 用于加载/保存 Scaler 的辅助函数 ---
def save_scaler(scaler: StandardScaler, path: str) -> None:
    """保存 StandardScaler 对象"""
    try:
        dump(scaler, path)
        print(f"Scaler 已保存到: {path}")
    except Exception as e:
        print(f"保存 Scaler 到 {path} 时出错: {e}")

def load_scaler(path: str) -> StandardScaler:
    """加载 StandardScaler 对象"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scaler 文件未找到: {path}. 可能需要先运行训练脚本。")
    try:
        scaler = load(path)
        print(f"Scaler 从 {path} 加载成功。")
        if not isinstance(scaler, StandardScaler):
             print(f"警告: 加载的对象类型不是 StandardScaler (实际类型: {type(scaler)})。")
        return scaler
    except Exception as e:
        print(f"加载 Scaler 从 {path} 时出错: {e}")
        raise

if __name__ == '__main__':
    # 测试数据工具的功能
    print("\n--- 测试数据工具 --- ")
    try:
        # 1. 加载配置
        config = LSTMConfig()
        
        # 2. 初始化处理器并预处理
        processor = NeuronDataProcessor(config)
        X_raw, y_raw = processor.preprocess_data()
        print(f"原始数据形状: X={X_raw.shape}, y={y_raw.shape}")
        
        # 3. 分割数据
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(X_raw, y_raw, config)
        
        # 4. 拟合和应用 Scaler
        processor.fit_scaler(X_train)
        X_train_scaled = processor.scale_data(X_train)
        X_val_scaled = processor.scale_data(X_val)
        X_test_scaled = processor.scale_data(X_test)
        print(f"标准化后形状: Train={X_train_scaled.shape}, Val={X_val_scaled.shape}, Test={X_test_scaled.shape}")
        
        # 5. 创建数据集
        train_dataset = NeuronDataset(X_train_scaled, y_train, config.sequence_length)
        val_dataset = NeuronDataset(X_val_scaled, y_val, config.sequence_length)
        test_dataset = NeuronDataset(X_test_scaled, y_test, config.sequence_length)
        print(f"数据集创建: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
        # 测试一下getitem
        seq, label = train_dataset[0]
        print(f"第一个训练样本: sequence shape={seq.shape}, label={label.item()}")
        
        # 6. 测试 Scaler 保存和加载
        save_scaler(processor.scaler, config.scaler_path)
        loaded_scaler = load_scaler(config.scaler_path)
        # 比较加载的 scaler 是否与原 scaler 参数一致 (可选)
        if isinstance(loaded_scaler, StandardScaler):
             print("Scaler 均值是否一致:", np.allclose(processor.scaler.mean_, loaded_scaler.mean_))
             print("Scaler 标准差是否一致:", np.allclose(processor.scaler.scale_, loaded_scaler.scale_))

        print("\n--- 数据工具测试完成 --- ")
        
    except Exception as e:
        print(f"\n--- 数据工具测试失败 --- ")
        import traceback
        traceback.print_exc() 