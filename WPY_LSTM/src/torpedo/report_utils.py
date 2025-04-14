import os
import json
import datetime
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List

# 尝试相对导入 config，如果此 utils 被其他 torpedo 模块调用
try:
    from .config import LSTMConfig
except ImportError:
    # 如果直接运行此脚本或从 src/ 运行，可能需要不同的导入方式
    # 但主要设计是在 train.py/evaluate_lstm.py 中调用，相对导入应优先
    print("报告工具: 无法进行相对导入 config，尝试直接导入 (可能在非预期环境下运行)")
    try:
        from config import LSTMConfig
    except ImportError:
        print("错误: 无法导入 LSTMConfig。请确保环境设置正确。")
        LSTMConfig = None # 设为 None 以允许脚本加载，但在调用时会失败

def get_report_path(config: LSTMConfig) -> str:
    """获取报告文件的标准路径"""
    return os.path.join(config.output_dir, f"summary_report_{config.data_identifier}.md")

def initialize_report(config: LSTMConfig):
    """创建或覆盖报告文件，并写入基本信息"""
    if not LSTMConfig:
        print("错误: LSTMConfig 未加载，无法初始化报告。")
        return
        
    report_path = get_report_path(config)
    print(f"--- 初始化报告文件: {report_path} ---")
    
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# LSTM 模型分析报告\n\n")
            f.write(f"**报告生成时间:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**数据标识符:** `{config.data_identifier}`\n")
            f.write(f"**数据文件路径:** `{config.data_file}`\n")
            f.write(f"**输出目录:** `{config.output_dir}`\n")
            f.write(f"**模型保存路径:** `{config.model_path}`\n")
            f.write(f"**Scaler 保存路径:** `{config.scaler_path}`\n")
            f.write("\n---\n\n")
            
            f.write(f"## 关键配置参数\n\n")
            f.write("| 参数 | 值 |\n")
            f.write("|---|---|\n")
            # 这里可以选择性地写入最重要的配置
            f.write(f"| `sequence_length` | {config.sequence_length} |\n")
            f.write(f"| `hidden_size` | {config.hidden_size} |\n")
            f.write(f"| `num_layers` | {config.num_layers} |\n")
            f.write(f"| `dropout` | {config.dropout} |\n")
            f.write(f"| `learning_rate` | {config.learning_rate} |\n")
            f.write(f"| `weight_decay` | {config.weight_decay} |\n")
            f.write(f"| `batch_size` | {config.batch_size} |\n")
            f.write(f"| `early_stopping_enabled` | {config.early_stopping_enabled} |\n")
            f.write(f"| `early_stopping_patience` | {config.early_stopping_patience} |\n")
            # f.write(f"| `reconstruction_loss_weight` | {config.reconstruction_loss_weight} |\n") # AE移除后可注释
            # 可以根据需要添加更多参数
            f.write("\n")
            
    except Exception as e:
        print(f"初始化报告时出错: {e}")

def append_training_summary(config: LSTMConfig, metrics: Dict[str, Any], total_time: float, stopped_early: bool, final_epoch: int):
    """追加训练摘要信息到报告"""
    if not LSTMConfig:
        print("错误: LSTMConfig 未加载，无法追加训练摘要。")
        return
        
    report_path = get_report_path(config)
    print(f"--- 向报告文件追加训练摘要: {report_path} ---")
    
    try:
        # 读取训练指标 CSV 获取更详细的信息
        metrics_df = pd.read_csv(config.metrics_log)
        best_val_loss_row = metrics_df.loc[metrics_df['val_loss'].idxmin()]
        best_val_acc_row = metrics_df.loc[metrics_df['val_acc'].idxmax()]
        
        with open(report_path, 'a', encoding='utf-8') as f:
            f.write(f"## 训练过程摘要\n\n")
            f.write(f"- **总训练时间:** {total_time:.2f} 秒\n")
            f.write(f"- **训练轮数:** {final_epoch} / {config.num_epochs}\n")
            stop_reason = "早停触发 (Early Stopping)" if stopped_early else "完成所有轮数 (Completed)"
            f.write(f"- **训练停止原因:** {stop_reason}\n")
            f.write(f"- **最佳验证损失 (Best Validation Loss):** {best_val_loss_row['val_loss']:.4f} (在 Epoch {int(best_val_loss_row['epoch'])})\n")
            f.write(f"- **最佳验证准确率 (Best Validation Accuracy):** {best_val_acc_row['val_acc']:.2f}% (在 Epoch {int(best_val_acc_row['epoch'])})\n")
            # 可以添加最后 epoch 的指标作为参考
            last_epoch_metrics = metrics_df.iloc[-1]
            f.write(f"- **最后 Epoch ({int(last_epoch_metrics['epoch'])}) 指标:** Train Loss: {last_epoch_metrics['train_loss']:.4f}, Train Acc: {last_epoch_metrics['train_acc']:.2f}%, Val Loss: {last_epoch_metrics['val_loss']:.4f}, Val Acc: {last_epoch_metrics['val_acc']:.2f}%\n")
            f.write(f"- **训练指标图:** `plots/training_metrics_{config.data_identifier}.png` (或 accuracy/loss 单独图)\n")
            f.write("\n---\n\n")
            
    except FileNotFoundError:
        print(f"错误: 找不到训练指标文件 {config.metrics_log}，无法追加训练摘要。")
    except Exception as e:
        print(f"追加训练摘要时出错: {e}")

def append_evaluation_summary(config: LSTMConfig):
    """追加评估摘要信息到报告"""
    if not LSTMConfig:
        print("错误: LSTMConfig 未加载，无法追加评估摘要。")
        return
        
    report_path = get_report_path(config)
    eval_json_path = config.eval_results_json
    print(f"--- 向报告文件追加评估摘要: {report_path} (来自 {eval_json_path}) ---")
    
    try:
        with open(eval_json_path, 'r') as f:
            results_data = json.load(f)
            
        with open(report_path, 'a', encoding='utf-8') as f:
            f.write(f"## 测试集评估摘要\n\n")
            
            accuracy = results_data.get('test_set_accuracy', 'N/A')
            if isinstance(accuracy, (float, int)):
                f.write(f"- **总体准确率 (Overall Accuracy):** {accuracy*100:.2f}%\n")
            else:
                f.write(f"- **总体准确率 (Overall Accuracy):** {accuracy}\n")
                
            auc_roc = results_data.get('test_set_auc_roc_ovr_macro', 'N/A')
            avg_prec = results_data.get('test_set_average_precision_macro', 'N/A')
            
            f.write(f"- **AUC-ROC (Macro Avg, OvR, Present Classes):** {auc_roc:.4f}\n")
            f.write(f"- **平均精度 (Macro Avg, Present Classes):** {avg_prec:.4f}\n")
            f.write(f"- **混淆矩阵图:** `plots/test_confusion_matrix_{config.data_identifier}.png`\n")
            
            # 添加分类报告摘要
            report_dict = results_data.get('test_set_classification_report')
            if report_dict:
                f.write("\n### 分类报告 (测试集)\n\n")
                # 将字典转换为 DataFrame 便于格式化为 Markdown 表格
                df_report = pd.DataFrame(report_dict).transpose()
                # 选择性保留关键列并重命名
                df_report = df_report[['precision', 'recall', 'f1-score', 'support']].reset_index()
                df_report.rename(columns={'index': 'Class'}, inplace=True)
                # 格式化数值
                float_cols = df_report.select_dtypes(include=['float']).columns
                format_dict = {col: '{:.3f}' for col in float_cols if col != 'support'}
                format_dict['support'] = '{:.0f}'
                # 生成 Markdown 表格
                f.write(df_report.to_markdown(index=False, floatfmt=".3f"))
                f.write("\n")
                
            # 提示测试集类别缺失（如果信息存在）
            # (这部分信息目前在评估脚本的日志中，不在JSON里，可以考虑未来加入JSON)
            # f.write("\n*注意：评估日志可能包含关于测试集中缺失类别的警告。*\n")
            
            f.write("\n---\n\n")
            
    except FileNotFoundError:
        print(f"错误: 找不到评估结果文件 {eval_json_path}，无法追加评估摘要。")
    except Exception as e:
        print(f"追加评估摘要时出错: {e}")

def append_kfold_summary(config: LSTMConfig, fold_accuracies: List[float], total_duration: float):
    """追加 K-Fold 交叉验证摘要信息到报告，并包含关键配置"""
    if not LSTMConfig:
        print("错误: LSTMConfig 未加载，无法追加 K-Fold 摘要。")
        return
        
    report_path = get_report_path(config)
    print(f"--- 向报告文件追加 K-Fold 摘要: {report_path} ---")
    
    try:
        mean_acc = np.mean(fold_accuracies)
        std_acc = np.std(fold_accuracies)
        k = len(fold_accuracies)
        
        with open(report_path, 'a', encoding='utf-8') as f:
            f.write(f"## {k}-Fold Stratified Cross-Validation 摘要\n\n")
            
            # --- 添加关键配置信息 ---
            f.write("### 本次 K-Fold 运行使用的关键配置\n\n")
            f.write("| 参数 | 值 |\n")
            f.write("|---|---|\n")
            f.write(f"| K-Folds | {k} (目标 {config.k_folds}) |\n") # 记录实际运行的 k
            f.write(f"| sequence_length | {config.sequence_length} |\n")
            f.write(f"| hidden_size | {config.hidden_size} |\n")
            f.write(f"| num_layers | {config.num_layers} |\n")
            f.write(f"| learning_rate | {config.learning_rate} |\n")
            f.write(f"| weight_decay | {config.weight_decay} |\n")
            f.write(f"| lr_scheduler_patience | {config.lr_scheduler_patience} |\n")
            f.write(f"| dropout | {config.dropout} |\n")
            f.write(f"| batch_size | {config.batch_size} |\n")
            # 可以尝试添加关于数据处理策略的推断 (如果 config 中有明确标记更好)
            # f.write(f"| 分割策略 | {'使用原始片段' if config.max_episode_len is None else f'分割片段 (max_len={config.max_episode_len})'} |\n") # 需要 max_episode_len
            # f.write(f"| 分层策略 | {'按时长'} (假设) |\n") # 假设是时长分层
            # f.write(f"| 边界过滤 | {'是'} (假设) |\n") # 假设开启
            # f.write(f"| 类别权重 | {'是'} (假设) |\n") # 假设开启
            f.write("\n*注意: 上述策略(分割/分层/过滤/权重)为代码当前实现状态的假设，建议在代码或配置中添加明确标记以获得更准确的报告。*\n\n")
            # ------------------------
            
            f.write(f"### 结果\n\n")
            f.write(f"- **交叉验证总耗时:** {total_duration:.2f} 秒\n")
            f.write(f"- **平均最佳验证准确率:** {mean_acc:.2f}% (+/- {std_acc:.2f}%)\n")
            f.write("- **各折最佳验证准确率:**\n")
            for i, acc in enumerate(fold_accuracies):
                f.write(f"  - Fold {i+1}: {acc:.2f}%\n")
            f.write("\n*注意: 此评估基于模型在交叉验证过程中在各自验证折上的最佳表现。*\n")
            f.write("\n---\n\n")
            
    except Exception as e:
        print(f"追加 K-Fold 摘要时出错: {e}")
        import traceback
        traceback.print_exc()

def finalize_report(config: LSTMConfig):
    """添加报告结尾"""
    if not LSTMConfig:
        return
        
    report_path = get_report_path(config)
    try:
        with open(report_path, 'a', encoding='utf-8') as f:
            f.write("## 报告结束\n")
            f.write(f"详细日志和图表请参见 `{os.path.relpath(config.output_dir, os.path.dirname(report_path))}` 目录。\n")
        print(f"--- 报告已终结: {report_path} ---")
    except Exception as e:
        print(f"终结报告时出错: {e}")

if __name__ == '__main__':
    # 用于测试报告生成工具
    print("--- 测试报告生成工具 --- ")
    if not LSTMConfig:
        print("无法执行测试，LSTMConfig 未加载。")
    else:
        try:
            test_config = LSTMConfig()
            print(f"使用配置: {test_config.data_identifier}")
            
            # 模拟生成报告流程
            initialize_report(test_config)
            
            # 模拟训练指标 (需要确保 metrics_log 存在或创建假的)
            # 假设 metrics_log 存在且有内容
            if os.path.exists(test_config.metrics_log):
                dummy_metrics = {'dummy': []} # 实际 metrics 由调用者提供
                append_training_summary(test_config, dummy_metrics, 123.45, True, 50)
            else:
                 print(f"跳过训练摘要追加，因为 {test_config.metrics_log} 不存在。")
                 
            # 模拟评估结果 (需要确保 eval_results_json 存在或创建假的)
            if os.path.exists(test_config.eval_results_json):
                 append_evaluation_summary(test_config)
            else:
                 print(f"跳过评估摘要追加，因为 {test_config.eval_results_json} 不存在。")
                 
            finalize_report(test_config)
            
            print("\n--- 测试完成 --- ")
            print(f"请检查生成的报告文件: {get_report_path(test_config)}")
            
        except Exception as e:
            print(f"测试报告工具时发生错误: {e}")
            import traceback
            traceback.print_exc() 