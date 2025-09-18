#!/usr/bin/env python3
"""
批量实验脚本 - 自动化进行多组GCN实验
根据README要求进行50-100组实验，收集各项指标
"""

import os
import json
import time
import datetime
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import subprocess
import sys

# 实验配置
MODELS = ['gcn', 'gat', 'sage', 'hybrid', 'gin', 'transformer', 'chebnet', 'ensemble']
WINDOW_SIZES = [20, 50, 100]
DEFAULT_RUNS = 50  # 默认实验组数
DATA_ROOT = "../../data"

class BatchExperimentRunner:
    """批量实验运行器"""
    
    def __init__(self, output_dir="batch_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
        
    def run_single_model_experiment(self, model_name, window_size, runs=50):
        """
        运行单个模型的实验
        
        Args:
            model_name: 模型名称 (gcn, gat, hybrid)
            window_size: 窗口大小 (20, 50, 100)
            runs: 实验运行次数
            
        Returns:
            dict: 实验结果摘要
        """
        print(f"\n{'='*60}")
        print(f"开始实验: {model_name.upper()} - Window Size: {window_size}")
        print(f"实验次数: {runs}")
        print(f"{'='*60}")
        
        # 构建命令
        cmd = [
            sys.executable, "run.py",
            "--model", model_name,
            "--runs", str(runs),
            "--dataset", DATA_ROOT,
            "--window_size", str(window_size)
        ]
        
        # 记录开始时间
        start_time = time.time()
        
        try:
            # 运行实验
            # 直接继承父进程的stdout/stderr，实时打印子进程日志，避免看起来“卡住”
            result = subprocess.run(
                cmd,
                cwd=Path.cwd(),
                timeout=7200  # 2小时超时
            )
            
            if result.returncode != 0:
                print(f"实验失败: {result.stderr}")
                return None
                
            # 计算总运行时间
            total_run_time = time.time() - start_time
            
            # 查找最新的结果文件
            result_dir = Path(f"result/{model_name}")
            if not result_dir.exists():
                print(f"结果目录不存在: {result_dir}")
                return None
                
            # 找到最新的摘要文件
            summary_files = list(result_dir.glob("summary_*.json"))
            if not summary_files:
                print(f"未找到摘要文件在: {result_dir}")
                return None
                
            latest_summary = max(summary_files, key=lambda x: x.stat().st_mtime)
            
            # 读取结果
            with open(latest_summary, 'r', encoding='utf-8') as f:
                summary_data = json.load(f)
            
            # 提取关键指标
            metrics = summary_data.get('metrics_summary', {})
            
            experiment_result = {
                'model': model_name,
                'window_size': window_size,
                'runs': runs,
                'timestamp': datetime.datetime.now().isoformat(),
                'total_run_time': total_run_time,
                'training_time_mean': metrics.get('training_time', {}).get('mean', 0),
                'accuracy_mean': metrics.get('accuracy', {}).get('mean', 0),
                'accuracy_std': metrics.get('accuracy', {}).get('std', 0),
                'precision_mean': metrics.get('precision', {}).get('mean', 0),
                'precision_std': metrics.get('precision', {}).get('std', 0),
                'recall_mean': metrics.get('recall', {}).get('mean', 0),
                'recall_std': metrics.get('recall', {}).get('std', 0),
                'f1_mean': metrics.get('f1', {}).get('mean', 0),
                'f1_std': metrics.get('f1', {}).get('std', 0),
                'summary_file': str(latest_summary)
            }
            
            print(f"实验完成!")
            print(f"平均准确率: {experiment_result['accuracy_mean']:.4f} ± {experiment_result['accuracy_std']:.4f}")
            print(f"平均F1分数: {experiment_result['f1_mean']:.4f} ± {experiment_result['f1_std']:.4f}")
            print(f"平均训练时间: {experiment_result['training_time_mean']:.2f}秒")
            print(f"总运行时间: {experiment_result['total_run_time']:.2f}秒")
            
            return experiment_result
            
        except subprocess.TimeoutExpired:
            print(f"实验超时: {model_name} - Window Size: {window_size}")
            return None
        except Exception as e:
            print(f"实验出错: {e}")
            return None
    
    def run_all_experiments(self, runs=50, models=None, window_sizes=None):
        """
        运行所有实验组合
        
        Args:
            runs: 每个实验的运行次数
            models: 要测试的模型列表，None表示使用默认所有模型
            window_sizes: 要测试的窗口大小列表，None表示使用默认所有窗口
        """
        if models is None:
            models = MODELS
        if window_sizes is None:
            window_sizes = WINDOW_SIZES
            
        total_experiments = len(models) * len(window_sizes)
        current_experiment = 0
        
        print(f"开始批量实验")
        print(f"模型: {models}")
        print(f"窗口大小: {window_sizes}")
        print(f"每个实验运行次数: {runs}")
        print(f"总实验数: {total_experiments}")
        
        batch_start_time = time.time()
        
        for model in models:
            for window_size in window_sizes:
                current_experiment += 1
                print(f"\n进度: {current_experiment}/{total_experiments}")
                
                result = self.run_single_model_experiment(model, window_size, runs)
                if result:
                    self.results.append(result)
                    # 实时保存结果
                    self.save_results()
        
        batch_total_time = time.time() - batch_start_time
        
        print(f"\n{'='*60}")
        print(f"批量实验完成!")
        print(f"成功完成: {len(self.results)}/{total_experiments} 个实验")
        print(f"总耗时: {batch_total_time/3600:.2f} 小时")
        print(f"{'='*60}")
        
        # 生成最终报告
        self.generate_report()
    
    def save_results(self):
        """保存实验结果"""
        # 保存为JSON格式
        json_file = self.output_dir / "batch_results.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        # 保存为CSV格式
        if self.results:
            df = pd.DataFrame(self.results)
            csv_file = self.output_dir / "batch_results.csv"
            df.to_csv(csv_file, index=False, encoding='utf-8')
            
        print(f"结果已保存到: {self.output_dir}")
    
    def generate_report(self):
        """生成实验报告"""
        if not self.results:
            print("没有实验结果可生成报告")
            return
        
        df = pd.DataFrame(self.results)
        
        # 生成汇总统计
        report = {
            "batch_summary": {
                "total_experiments": len(self.results),
                "timestamp": datetime.datetime.now().isoformat(),
                "models_tested": df['model'].unique().tolist(),
                "window_sizes_tested": df['window_size'].unique().tolist()
            },
            "performance_summary": {}
        }
        
        # 按模型分组统计
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            
            report["performance_summary"][model] = {
                "experiments_count": len(model_data),
                "avg_accuracy": {
                    "mean": float(model_data['accuracy_mean'].mean()),
                    "std": float(model_data['accuracy_mean'].std()),
                    "min": float(model_data['accuracy_mean'].min()),
                    "max": float(model_data['accuracy_mean'].max())
                },
                "avg_f1": {
                    "mean": float(model_data['f1_mean'].mean()),
                    "std": float(model_data['f1_mean'].std()),
                    "min": float(model_data['f1_mean'].min()),
                    "max": float(model_data['f1_mean'].max())
                },
                "avg_training_time": {
                    "mean": float(model_data['training_time_mean'].mean()),
                    "std": float(model_data['training_time_mean'].std())
                },
                "avg_total_run_time": {
                    "mean": float(model_data['total_run_time'].mean()),
                    "std": float(model_data['total_run_time'].std())
                }
            }
        
        # 保存报告
        report_file = self.output_dir / "experiment_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 打印简要报告
        print(f"\n实验报告摘要:")
        print(f"{'模型':<10} {'准确率':<15} {'F1分数':<15} {'训练时间(s)':<15}")
        print("-" * 60)
        
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            acc_mean = model_data['accuracy_mean'].mean()
            f1_mean = model_data['f1_mean'].mean()
            time_mean = model_data['training_time_mean'].mean()
            
            print(f"{model:<10} {acc_mean:<15.4f} {f1_mean:<15.4f} {time_mean:<15.2f}")
        
        print(f"\n详细报告已保存到: {report_file}")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="批量GCN实验脚本")
    parser.add_argument('--runs', type=int, default=DEFAULT_RUNS,
                        help=f'每个实验的运行次数 (默认: {DEFAULT_RUNS})')
    parser.add_argument('--models', nargs='+', choices=MODELS, default=['gcn', 'gat', 'sage', 'gin'],
                        help=f'要测试的模型 (默认: gcn, gat, sage, gin)')
    parser.add_argument('--window_sizes', type=int, nargs='+', default=WINDOW_SIZES,
                        help=f'要测试的窗口大小 (默认: {WINDOW_SIZES})')
    parser.add_argument('--output_dir', type=str, default="batch_results",
                        help='结果输出目录 (默认: batch_results)')
    parser.add_argument('--quick_test', action='store_true',
                        help='快速测试模式 (每个实验只运行5次)')
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 快速测试模式
    if args.quick_test:
        args.runs = 5
        print("快速测试模式: 每个实验只运行5次")
    
    # 检查当前目录
    if not Path("run.py").exists():
        print("错误: 请在GCNs目录下运行此脚本")
        sys.exit(1)
    
    # 创建实验运行器
    runner = BatchExperimentRunner(args.output_dir)
    
    # 运行批量实验
    try:
        runner.run_all_experiments(
            runs=args.runs,
            models=args.models,
            window_sizes=args.window_sizes
        )
    except KeyboardInterrupt:
        print("\n实验被用户中断")
        runner.save_results()
        print("已保存当前结果")
    except Exception as e:
        print(f"批量实验出错: {e}")
        runner.save_results()
        print("已保存当前结果")

if __name__ == "__main__":
    main()