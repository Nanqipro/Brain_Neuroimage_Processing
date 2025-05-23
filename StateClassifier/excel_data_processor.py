#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Excel钙离子数据处理器

专门用于处理Excel格式的神经元钙离子浓度数据，支持从EMtrace01.xlsx等文件
加载和预处理钙信号数据，为后续相空间重构和图神经网络训练做准备。

作者: Clade 4  
日期: 2025年5月23日
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
import logging
from pathlib import Path
from scipy.stats import zscore
import warnings

from config import config

# 配置日志
logger = logging.getLogger(__name__)


class ExcelCalciumDataProcessor:
    """
    Excel钙离子数据处理器类
    
    用于读取和处理Excel格式的神经元钙离子浓度时间序列数据。
    数据格式要求：
    - 第一列为stamp（时间戳）
    - 其余列为n1, n2, n3...（神经元ID对应的钙离子浓度）
    """
    
    def __init__(self, file_path: str):
        """
        初始化数据处理器
        
        Parameters
        ----------
        file_path : str
            Excel文件路径
        """
        self.file_path = Path(file_path)
        self.raw_data = None
        self.neuron_data = None
        self.timestamps = None
        self.neuron_ids = None
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"找不到数据文件: {file_path}")
            
    def load_excel_data(self) -> pd.DataFrame:
        """
        从Excel文件加载钙离子数据
        
        Returns
        -------
        pd.DataFrame
            加载的原始数据
        """
        try:
            logger.info(f"正在加载Excel数据文件: {self.file_path}")
            
            # 读取Excel文件
            self.raw_data = pd.read_excel(self.file_path)
            
            # 验证数据格式
            if 'stamp' not in self.raw_data.columns:
                raise ValueError("Excel文件必须包含'stamp'列作为时间戳")
                
            # 获取神经元列（除stamp外的所有列）
            neuron_columns = [col for col in self.raw_data.columns if col != 'stamp']
            
            if len(neuron_columns) == 0:
                raise ValueError("Excel文件必须包含至少一个神经元数据列")
                
            # 验证神经元列命名格式
            invalid_columns = []
            for col in neuron_columns:
                if not col.startswith('n') or not col[1:].isdigit():
                    invalid_columns.append(col)
                    
            if invalid_columns:
                logger.warning(f"发现非标准神经元列名: {invalid_columns}")
                logger.warning("建议使用n1, n2, n3...格式命名神经元列")
            
            # 提取时间戳和神经元数据
            self.timestamps = self.raw_data['stamp'].values
            self.neuron_data = self.raw_data[neuron_columns].values  # shape: (time_points, neurons)
            self.neuron_ids = neuron_columns
            
            logger.info(f"✓ 成功加载数据: {len(self.timestamps)} 个时间点, {len(self.neuron_ids)} 个神经元")
            logger.info(f"神经元列: {self.neuron_ids[:10]}..." if len(self.neuron_ids) > 10 else f"神经元列: {self.neuron_ids}")
            
            return self.raw_data
            
        except Exception as e:
            logger.error(f"✗ 加载Excel数据失败: {e}")
            raise
    
    def preprocess_data(self, 
                       remove_nan: bool = True, 
                       z_score_normalize: bool = True,
                       outlier_threshold: float = 5.0) -> np.ndarray:
        """
        预处理钙离子数据
        
        Parameters
        ----------
        remove_nan : bool, default=True
            是否移除包含NaN的行
        z_score_normalize : bool, default=True  
            是否进行Z-score标准化
        outlier_threshold : float, default=5.0
            异常值检测阈值（标准差倍数）
            
        Returns
        -------
        np.ndarray
            预处理后的数据，shape: (time_points, neurons)
        """
        if self.neuron_data is None:
            raise ValueError("请先调用load_excel_data()加载数据")
            
        logger.info("开始预处理钙离子数据...")
        
        processed_data = self.neuron_data.copy()
        original_shape = processed_data.shape
        
        # 1. 处理缺失值
        if remove_nan:
            # 检查NaN值
            nan_mask = np.isnan(processed_data)
            if np.any(nan_mask):
                nan_count = np.sum(nan_mask)
                logger.warning(f"发现 {nan_count} 个NaN值")
                
                # 移除包含NaN的行
                valid_rows = ~np.any(nan_mask, axis=1)
                processed_data = processed_data[valid_rows]
                self.timestamps = self.timestamps[valid_rows]
                
                logger.info(f"移除NaN后数据形状: {processed_data.shape}")
        
        # 2. 异常值检测和处理
        if outlier_threshold > 0:
            outlier_count = 0
            for neuron_idx in range(processed_data.shape[1]):
                neuron_signal = processed_data[:, neuron_idx]
                z_scores = np.abs(zscore(neuron_signal))
                outliers = z_scores > outlier_threshold
                
                if np.any(outliers):
                    outlier_count += np.sum(outliers)
                    # 用中位数替换异常值
                    median_val = np.median(neuron_signal[~outliers])
                    processed_data[outliers, neuron_idx] = median_val
                    
            if outlier_count > 0:
                logger.warning(f"检测并处理了 {outlier_count} 个异常值")
        
        # 3. Z-score标准化
        if z_score_normalize:
            logger.info("执行Z-score标准化...")
            # 按神经元维度标准化（每个神经元单独标准化）
            for neuron_idx in range(processed_data.shape[1]):
                neuron_signal = processed_data[:, neuron_idx]
                processed_data[:, neuron_idx] = zscore(neuron_signal)
        
        logger.info(f"✓ 数据预处理完成: {original_shape} -> {processed_data.shape}")
        
        return processed_data
    
    def convert_to_scn_format(self, processed_data: np.ndarray) -> List[List[np.ndarray]]:
        """
        将Excel数据转换为SCN处理流程所需的格式
        
        原始SCN格式: F_set[cell_num][timeline] = signal_array
        新格式: 每个神经元的完整时间序列作为一个信号
        
        Parameters
        ----------
        processed_data : np.ndarray
            预处理后的数据，shape: (time_points, neurons)
            
        Returns
        -------
        List[List[np.ndarray]]
            SCN格式的数据: [neuron_idx][timeline_idx] = signal_segment
        """
        logger.info("转换数据为SCN处理格式...")
        
        num_time_points, num_neurons = processed_data.shape
        
        # 创建SCN格式的数据结构
        # 我们将每个神经元的时间序列分段作为不同的"细胞状态"
        F_set = []
        
        # 参数配置
        segment_length = min(config.TRAJECTORY_LENGTH * 2, num_time_points // 4)  # 确保有足够的数据点
        overlap_ratio = 0.5  # 段间重叠比例
        step_size = int(segment_length * (1 - overlap_ratio))
        
        logger.info(f"分段参数: 段长度={segment_length}, 步长={step_size}")
        
        for neuron_idx in range(min(num_neurons, config.MAX_CELLS_PROCESS)):
            neuron_signal = processed_data[:, neuron_idx]
            neuron_segments = []
            
            # 滑动窗口分段
            for start_idx in range(0, num_time_points - segment_length + 1, step_size):
                end_idx = start_idx + segment_length
                segment = neuron_signal[start_idx:end_idx]
                
                # 验证分段质量
                if len(segment) >= config.MIN_SIGNAL_LENGTH and not np.all(np.isnan(segment)):
                    neuron_segments.append(segment)
                
                # 限制每个神经元的段数以控制计算量
                if len(neuron_segments) >= 10:  # 最多10个时间段
                    break
            
            F_set.append(neuron_segments)
            
        logger.info(f"✓ 转换完成: {len(F_set)} 个神经元, 平均每个神经元 {np.mean([len(segments) for segments in F_set]):.1f} 个时间段")
        
        return F_set
    
    def get_data_info(self) -> Dict:
        """
        获取数据集信息
        
        Returns
        -------
        Dict
            包含数据集基本信息的字典
        """
        if self.raw_data is None:
            return {"error": "数据未加载"}
            
        info = {
            "文件路径": str(self.file_path),
            "时间点数量": len(self.timestamps),
            "神经元数量": len(self.neuron_ids),
            "数据形状": self.neuron_data.shape if self.neuron_data is not None else None,
            "时间戳范围": f"{self.timestamps.min()} - {self.timestamps.max()}",
            "神经元ID": self.neuron_ids[:5] + ["..."] if len(self.neuron_ids) > 5 else self.neuron_ids,
            "缺失值数量": np.sum(np.isnan(self.neuron_data)) if self.neuron_data is not None else 0,
        }
        
        # 数据统计信息
        if self.neuron_data is not None:
            info.update({
                "数据范围": f"{np.nanmin(self.neuron_data):.3f} - {np.nanmax(self.neuron_data):.3f}",
                "数据均值": f"{np.nanmean(self.neuron_data):.3f}",
                "数据标准差": f"{np.nanstd(self.neuron_data):.3f}"
            })
        
        return info
    
    def save_processed_data(self, processed_data: np.ndarray, output_path: str = None):
        """
        保存预处理后的数据
        
        Parameters
        ----------
        processed_data : np.ndarray
            预处理后的数据
        output_path : str, optional
            输出文件路径，默认为原文件名_processed.csv
        """
        if output_path is None:
            output_path = self.file_path.parent / f"{self.file_path.stem}_processed.csv"
        
        # 创建DataFrame
        df = pd.DataFrame(processed_data, columns=self.neuron_ids)
        df.insert(0, 'stamp', self.timestamps)
        
        # 保存为CSV
        df.to_csv(output_path, index=False)
        logger.info(f"✓ 预处理数据已保存到: {output_path}")


def load_excel_calcium_data(file_path: str) -> Tuple[List[List[np.ndarray]], Dict]:
    """
    便捷函数：从Excel文件加载和预处理钙离子数据
    
    Parameters
    ----------
    file_path : str
        Excel文件路径
        
    Returns
    -------
    Tuple[List[List[np.ndarray]], Dict]
        (SCN格式的数据, 数据信息字典)
    """
    processor = ExcelCalciumDataProcessor(file_path)
    
    # 加载数据
    processor.load_excel_data()
    
    # 预处理
    processed_data = processor.preprocess_data()
    
    # 转换格式
    scn_data = processor.convert_to_scn_format(processed_data)
    
    # 获取信息
    data_info = processor.get_data_info()
    
    return scn_data, data_info


if __name__ == "__main__":
    # 测试代码
    file_path = "datasets/EMtrace01.xlsx"
    
    if Path(file_path).exists():
        print("测试Excel钙离子数据处理器...")
        
        try:
            scn_data, info = load_excel_calcium_data(file_path)
            
            print("\n数据信息:")
            for key, value in info.items():
                print(f"  {key}: {value}")
                
            print(f"\nSCN格式数据:")
            print(f"  神经元数量: {len(scn_data)}")
            print(f"  时间段数量: {[len(segments) for segments in scn_data]}")
            
        except Exception as e:
            print(f"测试失败: {e}")
    else:
        print(f"测试文件不存在: {file_path}") 