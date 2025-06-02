#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强版神经元状态分析测试脚本

测试修改后的4种状态分析功能和新增的可视化功能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from State_analysis import EnhancedStateAnalyzer
import os

# 设置matplotlib字体配置以避免字体警告
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.family'] = ['sans-serif']

# 额外的字体配置
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['mathtext.default'] = 'regular'

def generate_test_data(n_neurons: int = 100, n_timepoints: int = 1000, n_states: int = 4) -> pd.DataFrame:
    """
    生成测试用的神经元钙离子浓度数据
    
    Parameters
    ----------
    n_neurons : int
        神经元数量
    n_timepoints : int
        时间点数量
    n_states : int
        状态数量
        
    Returns
    -------
    pd.DataFrame
        模拟的神经元数据
    """
    np.random.seed(42)
    
    data = {}
    
    # 为每种状态生成特征鲜明的数据
    neurons_per_state = n_neurons // n_states
    
    for state in range(n_states):
        start_idx = state * neurons_per_state
        end_idx = (state + 1) * neurons_per_state if state < n_states - 1 else n_neurons
        
        for neuron_idx in range(start_idx, end_idx):
            neuron_name = f'n{neuron_idx + 1}'
            
            # 根据状态生成不同特征的信号
            if state == 0:  # State I: 高频连续振荡状态
                # 高频正弦波 + 噪声
                t = np.linspace(0, 100, n_timepoints)
                signal = 2 * np.sin(2 * np.pi * 0.8 * t) + 0.5 * np.random.normal(0, 0.3, n_timepoints)
                signal = np.abs(signal)  # 确保非负值
                
            elif state == 1:  # State II: 规律性脉冲放电状态
                # 周期性脉冲
                signal = np.zeros(n_timepoints)
                pulse_interval = 50
                pulse_width = 5
                for i in range(0, n_timepoints, pulse_interval):
                    if i + pulse_width < n_timepoints:
                        signal[i:i+pulse_width] = 3 + 0.5 * np.random.normal(0, 0.2, pulse_width)
                signal += 0.1 * np.random.normal(0, 0.1, n_timepoints)
                
            elif state == 2:  # State III: 间歇性突发状态
                # 突发性活动
                signal = 0.2 * np.random.normal(0, 0.1, n_timepoints)
                burst_starts = np.random.choice(n_timepoints, size=5, replace=False)
                for start in burst_starts:
                    burst_length = min(30, n_timepoints - start)
                    signal[start:start+burst_length] += np.exp(-np.arange(burst_length)/10) * 4
                    
            else:  # State IV: 不规律波动状态
                # 随机波动
                signal = np.random.gamma(2, 0.5, n_timepoints)
                signal = signal + 0.5 * np.sin(2 * np.pi * np.random.uniform(0.1, 0.3) * np.arange(n_timepoints))
            
            # 确保信号非负且在合理范围内
            signal = np.clip(signal, 0, 10)
            data[neuron_name] = signal
    
    return pd.DataFrame(data)

def test_enhanced_analysis():
    """测试增强版神经元状态分析功能"""
    print("开始测试增强版神经元状态分析功能...")
    
    # 创建输出目录
    output_dir = '../results/test_enhanced_analysis/'
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成测试数据
    print("生成测试数据...")
    test_data = generate_test_data(n_neurons=80, n_timepoints=800, n_states=4)
    
    # 保存测试数据
    test_data_path = os.path.join(output_dir, 'test_neuron_data.xlsx')
    test_data.to_excel(test_data_path, index=False)
    print(f"测试数据已保存到: {test_data_path}")
    
    # 初始化分析器
    print("初始化增强版分析器...")
    analyzer = EnhancedStateAnalyzer(sampling_rate=4.8)
    
    try:
        # 提取特征
        print("提取综合特征...")
        features, feature_names, neuron_names = analyzer.extract_comprehensive_features(test_data)
        print(f"特征提取完成：{features.shape[0]} 个神经元，{features.shape[1]} 个特征")
        
        # 识别状态
        print("识别神经元状态...")
        labels = analyzer.identify_states_enhanced(features, method='ensemble', n_states=4)
        
        # 打印状态分布
        unique_states, counts = np.unique(labels, return_counts=True)
        print("状态分布：")
        for state, count in zip(unique_states, counts):
            print(f"  State {state+1}: {count} 个神经元")
        
        # 生成可视化
        print("生成增强版可视化...")
        analyzer.visualize_enhanced_states(test_data, labels, neuron_names, output_dir)
        
        # 保存结果
        print("保存分析结果...")
        results_path = os.path.join(output_dir, 'test_analysis_results.xlsx')
        analyzer.save_enhanced_results(test_data, labels, neuron_names, features, 
                                     feature_names, results_path)
        
        print(f"测试完成！所有结果已保存到: {output_dir}")
        
        # 打印生成的文件列表
        print("\n生成的文件包括：")
        for file in os.listdir(output_dir):
            if file.endswith(('.png', '.html', '.xlsx')):
                print(f"  - {file}")
                
    except Exception as e:
        print(f"测试过程中出现错误: {str(e)}")
        raise

if __name__ == "__main__":
    test_enhanced_analysis() 