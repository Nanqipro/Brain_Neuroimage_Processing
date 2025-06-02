#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
时间窗口神经元状态分析 - 使用示例

该示例展示如何使用修改后的State_analysis.py进行时间窗口动态状态分析
"""

import os
import sys
import pandas as pd
import numpy as np

# 添加路径以导入State_analysis模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from State_analysis import EnhancedStateAnalyzer

def create_sample_data():
    """创建示例数据用于演示"""
    print("创建示例神经元钙离子浓度数据...")
    
    # 模拟4.8Hz采样率，10分钟的数据
    sampling_rate = 4.8
    duration = 600  # 10分钟
    n_points = int(sampling_rate * duration)
    time = np.linspace(0, duration, n_points)
    
    # 创建5个虚拟神经元
    data = {'time': time}
    
    for i in range(5):
        neuron_name = f'n{i+1}'
        
        # 模拟不同的状态变化
        signal = np.zeros(n_points)
        
        # 将时间分成几段，每段有不同的活动模式
        segment_length = n_points // 4
        
        for segment in range(4):
            start_idx = segment * segment_length
            end_idx = start_idx + segment_length if segment < 3 else n_points
            segment_time = time[start_idx:end_idx]
            
            if segment == 0:  # State I: 高频振荡
                freq = 0.8 + i * 0.1
                signal[start_idx:end_idx] = 2.0 + 1.5 * np.sin(2 * np.pi * freq * segment_time) + \
                                          0.3 * np.random.normal(0, 1, len(segment_time))
            elif segment == 1:  # State II: 规律脉冲
                pulse_interval = 20 + i * 5  # 脉冲间隔
                for j in range(0, len(segment_time), pulse_interval):
                    if j + 5 < len(segment_time):
                        signal[start_idx + j:start_idx + j + 5] = 3.0 + 0.5 * np.random.normal(0, 1, 5)
                signal[start_idx:end_idx] += 0.5 + 0.2 * np.random.normal(0, 1, len(segment_time))
            elif segment == 2:  # State III: 间歇突发
                for j in range(0, len(segment_time), 60):
                    burst_length = 10 + np.random.randint(0, 10)
                    if j + burst_length < len(segment_time):
                        signal[start_idx + j:start_idx + j + burst_length] = \
                            4.0 + 2.0 * np.random.normal(0, 1, burst_length)
                signal[start_idx:end_idx] += 0.3 + 0.1 * np.random.normal(0, 1, len(segment_time))
            else:  # State IV: 不规律波动
                signal[start_idx:end_idx] = 1.0 + 0.8 * np.random.normal(0, 1, len(segment_time)) + \
                                          0.5 * np.sin(2 * np.pi * 0.1 * segment_time)
        
        data[neuron_name] = signal
    
    return pd.DataFrame(data)

def run_temporal_analysis_example():
    """运行时间窗口分析示例"""
    print("=== 时间窗口神经元状态分析示例 ===\n")
    
    # 1. 创建示例数据
    data = create_sample_data()
    print(f"✅ 创建了包含 {len(data)} 个时间点和 {len([col for col in data.columns if col.startswith('n')])} 个神经元的示例数据")
    
    # 2. 初始化分析器（时间窗口模式）
    analyzer = EnhancedStateAnalyzer(
        sampling_rate=4.8,
        window_duration=60.0,    # 60秒窗口
        overlap_ratio=0.5        # 50%重叠
    )
    print(f"✅ 初始化时间窗口分析器 - 窗口长度: 60秒, 重叠率: 50%")
    
    # 3. 进行时间窗口状态分析
    print("\n📊 开始时间窗口状态分析...")
    labels, results_df = analyzer.analyze_temporal_states(
        data, 
        method='ensemble', 
        n_states=4
    )
    
    # 4. 显示分析结果
    print(f"\n🎯 分析结果摘要:")
    print(f"   - 总时间窗口数: {len(results_df)}")
    print(f"   - 分析神经元数: {results_df['neuron_id'].nunique()}")
    print(f"   - 识别状态数: {results_df['state_label'].nunique()}")
    
    # 显示每个神经元的状态多样性
    neuron_diversity = results_df.groupby('neuron_id')['state_label'].nunique()
    print(f"   - 神经元状态多样性: 平均 {neuron_diversity.mean():.2f} 种状态/神经元")
    
    # 显示状态分布
    print(f"\n📈 状态分布:")
    state_counts = results_df['state_name'].value_counts()
    for state, count in state_counts.items():
        percentage = count / len(results_df) * 100
        print(f"   - {state}: {count} 窗口 ({percentage:.1f}%)")
    
    # 显示状态转换统计
    print(f"\n🔄 状态转换分析:")
    transitions = {}
    for neuron in results_df['neuron_id'].unique():
        neuron_data = results_df[results_df['neuron_id'] == neuron].sort_values('window_idx')
        states = neuron_data['state_label'].values
        
        for i in range(len(states) - 1):
            transition = (states[i], states[i + 1])
            transitions[transition] = transitions.get(transition, 0) + 1
    
    # 显示主要转换
    sorted_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)[:5]
    for (from_state, to_state), count in sorted_transitions:
        print(f"   - State {from_state+1} → State {to_state+1}: {count} 次")
    
    # 5. 生成可视化
    output_dir = "../results/example_temporal_analysis/"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n🎨 生成可视化图表...")
    analyzer.visualize_temporal_states(data, results_df, output_dir)
    
    # 6. 保存结果
    features, feature_names, _ = analyzer.extract_windowed_features(data)
    output_file = os.path.join(output_dir, 'example_temporal_analysis_results.xlsx')
    analyzer.save_temporal_results(results_df, features, feature_names, output_file)
    
    print(f"✅ 结果已保存到: {output_file}")
    print(f"📊 可视化图表已保存到: {output_dir}")
    
    return results_df

def run_comparison_example():
    """运行对比示例：时间窗口 vs 传统分析"""
    print("\n=== 对比分析示例：时间窗口 vs 传统分析 ===\n")
    
    # 创建示例数据
    data = create_sample_data()
    
    # 1. 时间窗口分析
    print("🔍 进行时间窗口分析...")
    temporal_analyzer = EnhancedStateAnalyzer(
        sampling_rate=4.8,
        window_duration=60.0,
        overlap_ratio=0.5
    )
    
    labels_temporal, results_temporal = temporal_analyzer.analyze_temporal_states(
        data, method='ensemble', n_states=4
    )
    
    # 2. 传统分析（整个信号）
    print("🔍 进行传统全信号分析...")
    traditional_analyzer = EnhancedStateAnalyzer(sampling_rate=4.8)
    
    features_traditional, feature_names, neuron_names = \
        traditional_analyzer.extract_comprehensive_features_traditional(data)
    
    labels_traditional = traditional_analyzer.identify_states_enhanced(
        features_traditional, method='ensemble', n_states=4
    )
    
    # 3. 对比结果
    print(f"\n📊 对比结果:")
    print(f"   时间窗口分析:")
    print(f"   - 总分析单元: {len(results_temporal)} 个时间窗口")
    print(f"   - 状态多样性: {results_temporal.groupby('neuron_id')['state_label'].nunique().mean():.2f} 种状态/神经元")
    print(f"   - 状态切换检测: ✅ (可以检测时间变化)")
    
    print(f"\n   传统分析:")
    print(f"   - 总分析单元: {len(neuron_names)} 个神经元")
    print(f"   - 状态多样性: 每个神经元固定 1 种状态")
    print(f"   - 状态切换检测: ❌ (无法检测时间变化)")
    
    print(f"\n💡 主要差异:")
    print(f"   - 时间窗口分析能够捕捉神经元在不同时间段的状态变化")
    print(f"   - 传统分析只能为每个神经元分配一个固定状态")
    print(f"   - 时间窗口分析提供了状态转换和时间动态信息")
    
    return results_temporal

def main():
    """主函数"""
    print("🧠 神经元状态分析 - 时间窗口功能演示\n")
    
    try:
        # 运行时间窗口分析示例
        results_temporal = run_temporal_analysis_example()
        
        # 运行对比示例
        run_comparison_example()
        
        print(f"\n🎉 示例演示完成！")
        print(f"📋 主要功能:")
        print(f"   ✅ 时间窗口分割")
        print(f"   ✅ 窗口级特征提取")
        print(f"   ✅ 动态状态识别")
        print(f"   ✅ 状态转换分析")
        print(f"   ✅ 时间线可视化")
        print(f"   ✅ 状态多样性分析")
        
        print(f"\n💡 使用提示:")
        print(f"   - 通过 --window-duration 调整时间窗口长度")
        print(f"   - 通过 --overlap-ratio 调整窗口重叠程度")
        print(f"   - 通过 --analysis-mode temporal 启用时间窗口分析")
        print(f"   - 通过 --analysis-mode traditional 使用传统分析")
        
    except Exception as e:
        print(f"❌ 示例运行失败: {str(e)}")
        raise

if __name__ == "__main__":
    main() 