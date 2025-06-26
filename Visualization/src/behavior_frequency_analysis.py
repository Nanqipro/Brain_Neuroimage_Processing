#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按行为统计钙波频率的示例脚本

此脚本演示如何使用新的 analyze_behavior_total_calcium_frequency 函数
来统计不同行为状态下所有神经元的钙波总次数和频率。

作者: Assistant
日期: 2024
"""

import pandas as pd
import sys
import os

# 添加当前目录到Python路径，以便导入element_extraction模块
sys.path.append(os.path.dirname(__file__))
from element_extraction import analyze_behavior_total_calcium_frequency

def main():
    """
    主函数：演示按行为统计钙波频率的功能
    """
    
    # 数据文件路径
    data_file = "../datasets/processed_EMtrace01.xlsx"  # 请根据实际情况修改路径
    
    # 检查文件是否存在
    if not os.path.exists(data_file):
        print(f"错误: 找不到数据文件 '{data_file}'")
        print("请修改 data_file 变量为正确的文件路径")
        return
    
    try:
        # 加载数据
        print(f"正在加载数据文件: {data_file}")
        df = pd.read_excel(data_file)
        df.columns = [col.strip() for col in df.columns]  # 清理列名
        print(f"成功加载数据，共 {len(df)} 行")
        
        # 提取神经元列
        neuron_columns = [col for col in df.columns if col.startswith('n') and col[1:].isdigit()]
        print(f"检测到 {len(neuron_columns)} 个神经元: {neuron_columns[:5]}...")  # 只显示前5个
        
        # 检查是否存在行为标签列
        behavior_col = 'behavior'
        if behavior_col not in df.columns:
            print(f"错误: 数据中不包含行为标签列 '{behavior_col}'")
            print(f"可用的列: {list(df.columns)}")
            return
        
        # 显示行为标签信息
        behaviors = df[behavior_col].unique()
        print(f"发现 {len(behaviors)} 种行为标签: {behaviors}")
        
        # 执行按行为统计的钙波频率分析
        print("\n开始按行为统计钙波频率分析...")
        result_df = analyze_behavior_total_calcium_frequency(
            data_df=df,
            neuron_columns=neuron_columns,
            behavior_col=behavior_col,
            fs=0.7,  # 采样频率，请根据实际情况调整
            save_path="../results/behavior_total_frequency_analysis.csv",
            filter_strength=1.0,  # 过滤强度，可以调整
            adaptive_params=True
        )
        
        # 显示结果摘要
        print("\n=== 分析结果摘要 ===")
        print(result_df.to_string(index=False))
        
        # 计算一些统计信息
        if len(result_df) > 0:
            print("\n=== 统计信息 ===")
            total_row = result_df[result_df['behavior'] == 'ALL']
            if len(total_row) > 0:
                total_events = total_row.iloc[0]['total_calcium_events']
                total_time = total_row.iloc[0]['duration_seconds']
                total_freq = total_row.iloc[0]['frequency_hz']
                neuron_count = total_row.iloc[0]['neuron_count']
                
                print(f"总体统计:")
                print(f"  - 神经元数量: {neuron_count}")
                print(f"  - 总钙波事件数: {total_events}")
                print(f"  - 总时间: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
                print(f"  - 总频率: {total_freq:.4f} Hz")
                print(f"  - 平均每神经元频率: {total_freq/neuron_count:.4f} Hz")
            
            # 分行为统计
            behavior_rows = result_df[result_df['behavior'] != 'ALL']
            if len(behavior_rows) > 0:
                print(f"\n各行为统计:")
                for _, row in behavior_rows.iterrows():
                    behavior = row['behavior']
                    events = row['total_calcium_events']
                    duration = row['duration_seconds']
                    freq = row['frequency_hz']
                    avg_freq = row['avg_frequency_per_neuron']
                    
                    print(f"  - {behavior}: {events} 事件, {duration:.2f}秒, "
                          f"{freq:.4f} Hz (平均每神经元: {avg_freq:.4f} Hz)")
        
        print(f"\n结果已保存到: ../results/behavior_total_frequency_analysis.csv")
        
    except Exception as e:
        print(f"分析过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()

def compare_analysis_methods(data_file):
    """
    比较按神经元统计和按行为统计两种方法的结果
    
    参数
    ----------
    data_file : str
        数据文件路径
    """
    from element_extraction import analyze_behavior_calcium_frequency
    
    if not os.path.exists(data_file):
        print(f"错误: 找不到数据文件 '{data_file}'")
        return
    
    # 加载数据
    df = pd.read_excel(data_file)
    df.columns = [col.strip() for col in df.columns]
    neuron_columns = [col for col in df.columns if col.startswith('n') and col[1:].isdigit()]
    
    print("=== 方法比较 ===")
    
    # 方法1: 按神经元分别统计
    print("\n1. 按神经元分别统计频率:")
    neuron_freq_df = analyze_behavior_calcium_frequency(
        df, neuron_columns, behavior_col='behavior', fs=0.7
    )
    
    # 对按神经元的结果进行汇总
    behavior_summary = neuron_freq_df.groupby('behavior').agg({
        'calcium_events': 'sum',
        'duration_seconds': 'mean',
        'frequency_hz': 'sum'
    }).reset_index()
    
    print("按神经元统计后汇总的结果:")
    print(behavior_summary.to_string(index=False))
    
    # 方法2: 直接按行为统计
    print("\n2. 直接按行为统计总频率:")
    behavior_freq_df = analyze_behavior_total_calcium_frequency(
        df, neuron_columns, behavior_col='behavior', fs=0.7
    )
    
    print("直接按行为统计的结果:")
    print(behavior_freq_df.to_string(index=False))

if __name__ == "__main__":
    print("=== 按行为统计钙波频率分析工具 ===")
    print("此工具将统计每种行为状态下所有神经元的钙波总次数")
    print()
    
    # 运行主分析
    main()
    
    # 可选：运行比较分析
    # print("\n" + "="*50)
    # compare_analysis_methods("../datasets/processed_EMtrace01.xlsx") 