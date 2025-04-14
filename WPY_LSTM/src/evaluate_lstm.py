import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score, 
    roc_auc_score, 
    average_precision_score,
    precision_recall_curve, 
    roc_curve
)
from sklearn.preprocessing import label_binarize, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import os
import json
import datetime
import warnings
from typing import Tuple, List, Dict, Any
from analysis_utils import split_data # 导入新的分割函数
from joblib import load # 用于加载 scaler
from collections import Counter # 导入 Counter

# 从新的 torpedo 目录导入
from torpedo.config import LSTMConfig
from torpedo.data_utils import NeuronDataProcessor, NeuronDataset, split_data, load_scaler, find_behavior_episodes, get_indices_from_episodes
from torpedo.model import EnhancedNeuronLSTM, set_random_seed
from torpedo.report_utils import append_evaluation_summary, finalize_report # 导入报告工具

warnings.filterwarnings('ignore')

def load_data_model_and_scaler(config: LSTMConfig) -> Tuple[EnhancedNeuronLSTM, DataLoader, torch.device, Any, int]:
    """
    加载数据、模型、Scaler，并准备测试集。
    现在包含边界序列过滤和详细日志。
    """
    print("--- 加载数据、模型、Scaler和测试集 ---") 
    set_random_seed(config.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1. 加载和预处理数据 (获取未缩放 X, y)
    processor = NeuronDataProcessor(config)
    X, y = processor.preprocess_data() # 获取未缩放数据
    label_encoder = processor.label_encoder 
    num_classes = len(label_encoder.classes_)
    input_size = X.shape[1]
    class_names = list(label_encoder.classes_) # 获取类名用于日志
    print(f"数据加载完成。 输入维度: {input_size}, 类别数: {num_classes}, 类别名: {class_names}")

    # 2. 按时间顺序分割数据 (获取测试集数据)
    print("\n执行 Train/Validation/Test 数据分割 (按时间顺序)...")
    _, _, _, _, X_test, y_test = split_data(X, y, config) # 直接获取测试集数据
    print(f"原始测试数据提取完成: {X_test.shape}")
    # --- 日志: 原始测试集类别分布 ---
    original_test_counts = Counter(y_test)
    print(f"  原始测试集类别分布 (标签编码): {dict(original_test_counts)}")
    if original_test_counts:
        print(f"  原始测试集类别分布 (名称): {{ {', '.join([f'{label_encoder.inverse_transform([k])[0]}: {v}' for k, v in sorted(original_test_counts.items())])} }}")
    else:
        print("警告: 原始测试集标签 y_test 为空!")
    # -------------------------------

    # --- 获取测试集片段信息以进行边界过滤 ---
    print("\n识别测试集内部行为片段...")
    test_episodes = find_behavior_episodes(y_test, max_len=None)
    if not test_episodes:
        print("警告: 未在测试集中识别到片段，可能导致过滤后序列为空或无法进行边界检查。")
        test_indices_relative = np.array([], dtype=int)
    else:
        # test_indices_relative 是 episode 内各点相对于 X_test/y_test 起始点的索引
        test_indices_relative = get_indices_from_episodes(test_episodes) 
    print(f"识别得到 {len(test_episodes)} 个测试集片段。")
    # ------------------------------------------

    # 3. 加载 Scaler
    scaler = load_scaler(config.scaler_path)
    if not isinstance(scaler, StandardScaler):
        print("警告: 加载的对象不是 StandardScaler，可能导致错误。")

    # 4. 标准化测试数据
    print("\n--- 使用加载的 Scaler 标准化测试数据 ---")
    try:
        X_test_scaled = scaler.transform(X_test)
    except Exception as e:
        print(f"使用加载的 scaler 转换测试数据时出错: {e}")
        raise

    # 5. 创建初步测试数据集
    print("\n创建初步测试序列数据集...")
    initial_test_dataset = NeuronDataset(X_test_scaled, y_test, config.sequence_length)
    print(f"  初步测试数据集样本数 (可生成序列数): {len(initial_test_dataset)}")
    
    # --- 过滤测试集中的跨边界序列 ---
    print("开始过滤测试集中的跨边界序列...")
    num_test_before_filter = len(initial_test_dataset)
    valid_test_indices_for_subset = []
    discarded_sequence_labels = Counter() # 用于统计被丢弃序列的标签
    
    if not test_indices_relative.size: # 如果没有识别到片段或索引为空
        print("警告: test_indices_relative 为空，无法进行边界检查，将不过滤序列。")
        # 如果不过滤，所有索引都是有效的
        valid_test_indices_for_subset = list(range(num_test_before_filter))
    else:
        for j in range(num_test_before_filter): # j 是 initial_test_dataset 的索引
            seq_end_idx_in_subset = j + config.sequence_length - 1
            
            # 获取这个序列对应的原始 y_test 中的标签 (序列的最后一个点)
            sequence_label = -1 # 默认值或错误标记
            if seq_end_idx_in_subset < len(y_test):
                 sequence_label = y_test[seq_end_idx_in_subset] 
            else:
                 print(f"警告: 索引 {seq_end_idx_in_subset} 超出 y_test 范围 {len(y_test)}，跳过序列 {j}")
                 discarded_sequence_labels[-99] += 1 # 使用特殊键标记索引错误
                 continue
            
            # 检查 test_indices_relative 中的索引是否连续
            is_contiguous = True
            # 检查序列的第一个点 j 是否在 test_indices_relative 中有效
            if j >= len(test_indices_relative):
                is_contiguous = False # 序列的起始点已经超出了有效索引范围
            else:
                original_start_idx = test_indices_relative[j]
                for k in range(1, config.sequence_length):
                    current_relative_index_pos = j + k
                    if current_relative_index_pos >= len(test_indices_relative) or \
                       test_indices_relative[current_relative_index_pos] != original_start_idx + k:
                        is_contiguous = False
                        break
                        
            if is_contiguous:
                valid_test_indices_for_subset.append(j)
            else:
                # 记录被丢弃的序列的标签
                discarded_sequence_labels[sequence_label] += 1

    # --- 日志: 被丢弃序列的类别分布 ---
    if discarded_sequence_labels:
        print("  因跨越片段边界或索引错误而被丢弃的序列的类别分布 (标签编码):")
        print(f"    {dict(discarded_sequence_labels)}")
        print(f"  被丢弃序列的类别分布 (名称):")
        discarded_names = {}
        for k, v in sorted(discarded_sequence_labels.items()):
             try:
                 name = label_encoder.inverse_transform([k])[0] if k != -99 else "IndexError"
                 discarded_names[name] = v
             except ValueError: # 处理未知的标签编码
                 discarded_names[f"UnknownLabel({k})"] = v
        print(f"    {discarded_names}")
    else:
         print("  没有序列因跨边界而被丢弃。")
    # -----------------------------------
            
    filtered_test_dataset = Subset(initial_test_dataset, valid_test_indices_for_subset)
    num_test_after_filter = len(filtered_test_dataset)
    print(f"  最终测试集序列过滤: {num_test_before_filter} -> {num_test_after_filter}")
    # ----------------------------------

    # 6. 创建最终测试数据加载器 (使用过滤后的数据集)
    if num_test_after_filter == 0:
        print("警告: 过滤后测试集序列数量为零。评估将无法进行。")
        # 可以选择在这里抛出错误或返回特殊值
        # 返回空 loader 以便后续步骤可以检查
        test_loader = DataLoader([]) 
    else:
        test_loader = DataLoader(filtered_test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print(f"最终测试数据加载器创建完成 (过滤后)。")

    # 7. 初始化模型结构
    model = EnhancedNeuronLSTM(
        input_size=input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_classes=num_classes,
        latent_dim=config.latent_dim,
        num_heads=config.num_heads,
        dropout=config.dropout # 评估模式下 dropout 不生效，但结构需一致
    ).to(device)
    print(f"模型结构初始化完成。")

    # 8. 加载训练好的模型权重
    if not os.path.exists(config.model_path):
        raise FileNotFoundError(f"模型文件未找到: {config.model_path}")
    try:
        # 确保使用 weights_only=False (如果模型保存时包含非权重对象)
        checkpoint = torch.load(config.model_path, map_location=device, weights_only=False) 
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"从 checkpoint 加载模型状态字典成功。")
        else:
            model.load_state_dict(checkpoint)
            print(f"直接加载模型状态字典成功。")
    except Exception as e: print(f"加载模型权重时出错: {e}"); raise
    model.eval()
    print("模型加载完成并设置为评估模式。")
    
    return model, test_loader, device, label_encoder, num_classes

def evaluate_model(model: EnhancedNeuronLSTM, 
                   test_loader: DataLoader, 
                   device: torch.device, 
                   num_classes: int,
                   label_encoder: Any,
                   config: LSTMConfig) -> Dict[str, Any]:
    """
    在测试集上评估模型并计算详细指标。
    现在能处理空的 test_loader。
    """
    print("--- 开始在测试集上进行模型评估 ---")
    
    # 检查 test_loader 是否为空
    if not test_loader or not test_loader.dataset:
        print("错误: 测试数据加载器为空 (可能因序列过滤)，无法进行评估。")
        # 返回包含错误信息的字典
        return {
            'test_set_accuracy': 'N/A (No data)',
            'test_set_classification_report': {},
            'test_set_confusion_matrix': [],
            'test_set_auc_roc_ovr_macro': 'N/A (No data)',
            'test_set_average_precision_macro': 'N/A (No data)',
            'error': 'Test DataLoader is empty after filtering.'
        }

    all_y_true, all_y_pred, all_y_prob = [], [], []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs, _, _ = model(batch_X)
            probabilities = torch.softmax(outputs, dim=1)
            all_y_prob.append(probabilities.cpu().numpy())
            _, predicted = torch.max(outputs, 1)
            all_y_pred.append(predicted.cpu().numpy())
            all_y_true.append(batch_y.cpu().numpy())
    y_true, y_pred, y_prob = np.concatenate(all_y_true), np.concatenate(all_y_pred), np.concatenate(all_y_prob)
    print(f"模型在测试集上预测完成。样本总数: {len(y_true)}")
    class_names = label_encoder.classes_
    print("--- 计算测试集评估指标 ---")
    results = {}
    accuracy = accuracy_score(y_true, y_pred)
    results['test_set_accuracy'] = accuracy
    print(f"测试集总体准确率: {accuracy:.4f}")
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, labels=np.arange(num_classes), zero_division=0)
    results['test_set_classification_report'] = report
    print("\n测试集分类报告:")
    print(classification_report(y_true, y_pred, target_names=class_names, labels=np.arange(num_classes), zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    results['test_set_confusion_matrix'] = cm.tolist()
    print("\n测试集混淆矩阵生成完成。")
    y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
    valid_classes_roc = np.where(np.sum(y_true_bin, axis=0) > 0)[0]
    if len(valid_classes_roc) < num_classes:
         print(f"警告: 测试集中缺少 {num_classes - len(valid_classes_roc)} 个类别，无法为这些类别计算AUC-ROC。")
         if len(valid_classes_roc) > 1:
            try:
                 roc_auc = roc_auc_score(y_true_bin[:, valid_classes_roc], y_prob[:, valid_classes_roc], multi_class='ovr', average='macro')
                 results['test_set_auc_roc_ovr_macro'] = roc_auc
                 print(f"测试集 AUC-ROC (OvR, Macro Avg for present classes): {roc_auc:.4f}")
            except ValueError as e:
                 print(f"计算测试集 Macro Avg AUC-ROC 时出错: {e}")
                 results['test_set_auc_roc_ovr_macro'] = 'Error'
         else:
             print("测试集中出现的类别少于2个，无法计算 Macro Avg AUC-ROC。")
             results['test_set_auc_roc_ovr_macro'] = 'N/A'
    else:
        try:
            roc_auc = roc_auc_score(y_true_bin, y_prob, multi_class='ovr', average='macro')
            results['test_set_auc_roc_ovr_macro'] = roc_auc
            print(f"测试集 AUC-ROC (OvR, Macro Avg): {roc_auc:.4f}")
        except ValueError as e:
            print(f"计算测试集 Macro Avg AUC-ROC 时出错: {e}")
            results['test_set_auc_roc_ovr_macro'] = 'Error'

    avg_precision = {}
    present_classes = valid_classes_roc
    for i in present_classes:
        try: avg_precision[class_names[i]] = average_precision_score(y_true_bin[:, i], y_prob[:, i])
        except ValueError: avg_precision[class_names[i]] = np.nan
    results['test_set_average_precision_per_class'] = avg_precision
    macro_ap = np.nanmean([ap for ap in avg_precision.values() if not np.isnan(ap)])
    results['test_set_average_precision_macro'] = macro_ap
    print(f"测试集平均精确率 (Macro Avg for present classes): {macro_ap:.4f}")
    results['raw_test_predictions'] = {'y_true': y_true.tolist(), 'y_pred': y_pred.tolist(), 'y_prob': y_prob.tolist()}
    print("--- 测试集指标计算完成 ---")
    return results

def plot_confusion_matrix(cm: np.ndarray, 
                          class_names: List[str], 
                          save_path: str,
                          config: LSTMConfig) -> None:
    """
    绘制并保存混淆矩阵热图。
    """
    print(f"--- 绘制混淆矩阵并保存到: {save_path} ---")
    plt.style.use('seaborn-v0_8-whitegrid')
    figsize = (12, 10) # 可以从 config 读取
    fontsize = 10
    cmap = 'Blues'
    dpi = 300
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    plt.figure(figsize=figsize)
    try: sns.heatmap(df_cm, annot=True, fmt="d", cmap=cmap, linewidths=.5, cbar=False, annot_kws={"size": fontsize * 0.8})
    except ValueError: print("警告：无法生成混淆矩阵热图。"); plt.close(); return
    plt.title('Confusion Matrix (Test Set)', fontsize=fontsize * 1.4)
    plt.ylabel('True Label', fontsize=fontsize * 1.2)
    plt.xlabel('Predicted Label', fontsize=fontsize * 1.2)
    plt.xticks(rotation=45, ha='right', fontsize=fontsize)
    plt.yticks(rotation=0, fontsize=fontsize)
    plt.tight_layout()
    try: plt.savefig(save_path, dpi=dpi, bbox_inches='tight'); print(f"混淆矩阵图像已保存: {save_path}")
    except Exception as e: print(f"保存混淆矩阵图像时出错: {e}")
    plt.close()

def save_results(results: Dict[str, Any], config: LSTMConfig) -> None:
    """
    将评估结果保存到JSON文件。
    """
    save_path = config.eval_results_json # 直接使用config中的路径
    print(f"--- 保存测试集评估结果到: {save_path} ---")
    try:
        with open(save_path, 'w') as f: json.dump(results, f, indent=4)
        print("评估结果已成功保存。")
    except TypeError as e:
        print(f"保存结果时遇到TypeError: {e}")
        # 尝试保存部分可序列化的内容
        serializable_results = {}
        for key, value in results.items():
            try:
                json.dumps({key: value}) # 测试是否可序列化
                serializable_results[key] = value
            except TypeError:
                print(f"  - 跳过无法序列化的键: {key}")
        try:
            with open(save_path.replace('.json', '_partial.json'), 'w') as f:
                 json.dump(serializable_results, f, indent=4)
            print("部分可序列化的结果已保存。")
        except Exception as final_e:
             print(f"保存部分结果时仍发生错误: {final_e}")
             
    except Exception as e:
        print(f"保存评估结果时发生未知错误: {e}")

def main():
    """
    主执行函数。
    """
    print(f"测试集评估脚本启动于: {datetime.datetime.now()}") 
    config = None # 初始化
    evaluation_completed = False # 标记
    
    try:
        config = LSTMConfig()
        model, test_loader, device, label_encoder, num_classes = load_data_model_and_scaler(config)
        evaluation_metrics = evaluate_model(model, test_loader, device, num_classes, label_encoder, config)
        cm_np = np.array(evaluation_metrics['test_set_confusion_matrix'])
        plot_confusion_matrix(cm_np, list(label_encoder.classes_), config.confusion_matrix_plot, config)
        save_results(evaluation_metrics, config)
        evaluation_completed = True # 标记评估成功
        
        print(f"测试集评估脚本完成于: {datetime.datetime.now()}") 

    except FileNotFoundError as e:
        print(f"错误: 必要文件未找到 - {e}")
    except ImportError as e:
        print(f"错误: 模块导入失败 - {e}")
    except ValueError as e:
        print(f"错误: 数据或参数配置问题 - {e}")
    except RuntimeError as e:
        print(f"错误: 运行时错误 (可能是 CUDA 内存或计算问题) - {e}")
    except Exception as e:
        print(f"发生未知错误: {e}")
        if config:
            # 可以选择在错误日志中记录评估错误
            with open(config.error_log, 'a') as f:
                 import traceback
                 traceback.print_exc(file=f)
                 f.write(f"\n评估脚本在 {datetime.datetime.now()} 异常终止。\n")
        else:
             print("配置未加载，无法写入错误日志。")
             
    finally:
        # 无论是否成功，只要配置加载了，就尝试更新和终结报告
        if config:
            if evaluation_completed:
                # 追加评估摘要到报告
                append_evaluation_summary(config)
            # 终结报告
            finalize_report(config)
        else:
            print("配置未加载，无法生成或终结报告。")
        
if __name__ == "__main__":
    main() 