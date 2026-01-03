#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析多次训练实验结果
"""
import os
import re
import json
import pandas as pd
import numpy as np

def extract_results_from_log(log_file):
    """从日志文件提取汇总结果"""
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取汇总结果
        pattern = r'测试集准确率: ([\d.]+) ± ([\d.]+)\n测试集精确率: ([\d.]+) ± ([\d.]+)\n测试集召回率: ([\d.]+) ± ([\d.]+)\n测试集F1分数: ([\d.]+) ± ([\d.]+)'
        match = re.search(pattern, content)
        
        if match:
            return {
                'accuracy_mean': float(match.group(1)),
                'accuracy_std': float(match.group(2)),
                'precision_mean': float(match.group(3)),
                'precision_std': float(match.group(4)),
                'recall_mean': float(match.group(5)),
                'recall_std': float(match.group(6)),
                'f1_mean': float(match.group(7)),
                'f1_std': float(match.group(8))
            }
        return None
    except Exception as e:
        print(f"Error reading {log_file}: {e}")
        return None

def main():
    log_dir = "multi_run_logs_20251023_110254"
    
    # 定义数据集和模型
    datasets = [
        ('MUTAG', 'tudataset'),
        ('ogbg-molhiv', 'ogb'),
        ('PROTEINS', 'tudataset')
    ]
    
    models = ['gcn', 'gat', 'sage', 'hybrid', 'gin', 'chebnet', 'edgeconv', 'gunet', 'pna', 'gatv2', 'deepergcn']
    
    results = []
    
    # 提取所有结果
    for dataset_name, dataset_type in datasets:
        for model in models:
            log_file = f"{log_dir}/{dataset_name}_{model}_multi.log"
            
            if os.path.exists(log_file):
                result = extract_results_from_log(log_file)
                if result:
                    result['dataset'] = dataset_name
                    result['model'] = model
                    result['dataset_type'] = dataset_type
                    results.append(result)
                    print(f"✓ {dataset_name} + {model}")
                else:
                    print(f"✗ {dataset_name} + {model} - 无法提取结果")
            else:
                print(f"✗ {dataset_name} + {model} - 文件不存在")
    
    # 转换为DataFrame
    df = pd.DataFrame(results)
    
    # 保存为CSV
    df.to_csv('all_results_summary.csv', index=False)
    print(f"\n已保存 {len(results)} 个实验结果到 all_results_summary.csv")
    
    # 打印汇总统计
    print("\n" + "=" * 100)
    print("📊 实验结果汇总分析")
    print("=" * 100)
    
    for dataset_name, dataset_type in datasets:
        dataset_df = df[df['dataset'] == dataset_name]
        
        print(f"\n{'=' * 100}")
        print(f"数据集: {dataset_name}")
        print(f"{'=' * 100}")
        print(f"{'模型':<15} {'测试F1 (均值±标准差)':<25} {'测试准确率':<25} {'排名'}")
        print("-" * 100)
        
        # 按F1排序
        dataset_df_sorted = dataset_df.sort_values('f1_mean', ascending=False)
        
        for idx, (_, row) in enumerate(dataset_df_sorted.iterrows(), 1):
            model_name = row['model'].upper()
            f1_str = f"{row['f1_mean']:.4f} ± {row['f1_std']:.4f}"
            acc_str = f"{row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f}"
            rank = f"#{idx}"
            
            # 高亮最佳结果
            if idx == 1:
                print(f"🥇 {model_name:<13} {f1_str:<25} {acc_str:<25} {rank}")
            elif idx == 2:
                print(f"🥈 {model_name:<13} {f1_str:<25} {acc_str:<25} {rank}")
            elif idx == 3:
                print(f"🥉 {model_name:<13} {f1_str:<25} {acc_str:<25} {rank}")
            else:
                print(f"   {model_name:<13} {f1_str:<25} {acc_str:<25} {rank}")
    
    # 跨数据集模型性能对比
    print(f"\n{'=' * 100}")
    print("🏆 跨数据集模型性能排名 (按平均F1分数)")
    print(f"{'=' * 100}")
    
    model_avg_performance = df.groupby('model').agg({
        'f1_mean': 'mean',
        'accuracy_mean': 'mean',
        'f1_std': 'mean'
    }).sort_values('f1_mean', ascending=False)
    
    print(f"{'模型':<15} {'平均F1':<15} {'平均准确率':<15} {'平均稳定性(std)':<20} {'排名'}")
    print("-" * 100)
    
    for idx, (model, row) in enumerate(model_avg_performance.iterrows(), 1):
        model_name = model.upper()
        f1 = f"{row['f1_mean']:.4f}"
        acc = f"{row['accuracy_mean']:.4f}"
        std = f"{row['f1_std']:.4f}"
        rank = f"#{idx}"
        
        if idx == 1:
            print(f"🥇 {model_name:<13} {f1:<15} {acc:<15} {std:<20} {rank}")
        elif idx == 2:
            print(f"🥈 {model_name:<13} {f1:<15} {acc:<15} {std:<20} {rank}")
        elif idx == 3:
            print(f"🥉 {model_name:<13} {f1:<15} {acc:<15} {std:<20} {rank}")
        else:
            print(f"   {model_name:<13} {f1:<15} {acc:<15} {std:<20} {rank}")
    
    # 保存JSON格式汇总
    summary = {
        'total_experiments': len(results),
        'datasets': len(datasets),
        'models': len(models),
        'results': results,
        'model_rankings': model_avg_performance.to_dict('index')
    }
    
    with open('results_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'=' * 100}")
    print("✅ 分析完成！")
    print(f"   - CSV文件: all_results_summary.csv")
    print(f"   - JSON文件: results_analysis.json")
    print(f"{'=' * 100}\n")

if __name__ == '__main__':
    main()

