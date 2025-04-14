import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
import datetime
import warnings
from typing import Dict, Any, Tuple, List, Optional
from sklearn.preprocessing import StandardScaler
import math # 需要 math.ceil 或整数除法技巧
# 导入计算类别权重的工具
from sklearn.utils.class_weight import compute_class_weight 

# 从同级目录导入配置、数据工具和模型 (使用相对导入)
from .config import LSTMConfig
from .data_utils import NeuronDataProcessor, NeuronDataset, split_data, save_scaler, find_behavior_episodes, get_indices_from_episodes
from .model import EnhancedNeuronLSTM, set_random_seed
from .report_utils import initialize_report, append_training_summary, append_kfold_summary

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
                # 或者: num_sub_episodes = (episode_len + max_len - 1) // max_len
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

# --- Helper function for custom stratified K-Fold by episode ---
def stratified_kfold_by_episode(episodes: List[Tuple[int, int, int]],
                                k: int,
                                random_state: Optional[int] = None
                               ) -> List[List[Tuple[int, int, int]]]:
    """
    Performs stratified K-Fold splitting based on behavior episodes.
    Tries to balance the *total length (number of samples)* of episodes
    per class across folds.

    Args:
        episodes: List of (start_index, end_index, label) tuples.
        k: Number of folds.
        random_state: Seed for shuffling episodes within each class.

    Returns:
        A list of K lists. Each inner list contains the episode tuples
        assigned to that validation fold.
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Group episodes by label and calculate lengths
    episodes_by_label: Dict[int, List[Tuple[int, int, int, int]]] = {} # Store length too
    for start, end, label in episodes:
        length = end - start + 1
        if label not in episodes_by_label:
            episodes_by_label[label] = []
        episodes_by_label[label].append((start, end, label, length))

    # Initialize folds (each fold will store validation episodes)
    folds_val_episodes: List[List[Tuple[int, int, int]]] = [[] for _ in range(k)]
    # Keep track of total length per fold per class (optional, for balancing heuristic)
    # For simplicity, we'll use a greedy approach based on sorting by length.

    # Distribute episodes for each class across folds, aiming for length balance
    for label, label_episodes_with_len in episodes_by_label.items():
        # Sort episodes by length (descending) to prioritize assigning long ones first
        label_episodes_with_len.sort(key=lambda x: x[3], reverse=True)
        # Keep track of current total length in each fold for this class
        fold_lengths = [0] * k

        for start, end, ep_label, length in label_episodes_with_len:
            # Find the fold with the minimum current total length for this class
            target_fold_idx = np.argmin(fold_lengths)
            # Assign the episode (without length) to the target fold
            folds_val_episodes[target_fold_idx].append((start, end, ep_label))
            # Update the total length for that fold
            fold_lengths[target_fold_idx] += length

    # Optional: Shuffle the episodes within each validation fold
    if random_state is not None:
        for fold_list in folds_val_episodes:
            np.random.shuffle(fold_list)

    return folds_val_episodes

# --- Helper function to get data indices from episodes ---
def get_indices_from_episodes(episode_list: List[Tuple[int, int, int]]) -> np.ndarray:
    """Converts a list of episodes to a flat array of data indices."""
    indices = []
    for start, end, _ in episode_list:
        indices.extend(range(start, end + 1))
    return np.array(indices, dtype=int)

# --- 绘图函数 ---
def plot_training_metrics(metrics: Dict[str, List], config: LSTMConfig):
    """
    绘制训练和验证指标的变化曲线
    """
    print("--- 绘制训练指标图 --- ")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, len(metrics['train_losses']) + 1)

    # 损失曲线
    axes[0].plot(epochs, metrics['train_losses'], label='Training Loss', linewidth=2)
    axes[0].plot(epochs, metrics['val_losses'], label='Validation Loss', linewidth=2)
    axes[0].set_title('Loss Curves')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # 准确率曲线
    axes[1].plot(epochs, metrics['train_accuracies'], label='Training Accuracy', linewidth=2)
    axes[1].plot(epochs, metrics['val_accuracies'], label='Validation Accuracy', linewidth=2)
    axes[1].set_title('Accuracy Curves')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True)
    
    # 重构损失曲线
    axes[2].plot(epochs, metrics['reconstruction_losses'], label='Reconstruction Loss', linewidth=2)
    axes[2].set_title('Reconstruction Loss')
    axes[2].set_xlabel('Epochs')
    axes[2].set_ylabel('Loss')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    # 保存合并的图
    combined_plot_path = os.path.join(config.plot_dir, f"training_metrics_{config.data_identifier}.png")
    try:
        plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
        print(f"训练指标图已保存到: {combined_plot_path}")
    except Exception as e:
        print(f"保存训练指标图时出错: {e}")
    # 同时保存单独的准确率和损失图 (使用 config 中定义的路径)
    try:
         # 准确率图
         fig_acc, ax_acc = plt.subplots(1, 1, figsize=(7, 5))
         ax_acc.plot(epochs, metrics['train_accuracies'], label='Training Accuracy', linewidth=2)
         ax_acc.plot(epochs, metrics['val_accuracies'], label='Validation Accuracy', linewidth=2)
         ax_acc.set_title('Accuracy Curves')
         ax_acc.set_xlabel('Epochs')
         ax_acc.set_ylabel('Accuracy (%)')
         ax_acc.legend()
         ax_acc.grid(True)
         plt.tight_layout()
         plt.savefig(config.accuracy_plot, dpi=300, bbox_inches='tight')
         plt.close(fig_acc)
         print(f"准确率曲线图已保存到: {config.accuracy_plot}")
         
         # 损失图
         fig_loss, ax_loss = plt.subplots(1, 1, figsize=(7, 5))
         ax_loss.plot(epochs, metrics['train_losses'], label='Training Loss', linewidth=2)
         ax_loss.plot(epochs, metrics['val_losses'], label='Validation Loss', linewidth=2)
         ax_loss.plot(epochs, metrics['reconstruction_losses'], label='Reconstruction Loss', linestyle='--', linewidth=1.5)
         ax_loss.set_title('Loss Curves')
         ax_loss.set_xlabel('Epochs')
         ax_loss.set_ylabel('Loss')
         ax_loss.legend()
         ax_loss.grid(True)
         plt.tight_layout()
         plt.savefig(config.loss_plot, dpi=300, bbox_inches='tight')
         plt.close(fig_loss)
         print(f"损失曲线图已保存到: {config.loss_plot}")
         
    except Exception as e:
         print(f"保存单独图表时出错: {e}")
    plt.close(fig) # 关闭合并图

# --- 训练函数 ---
def run_training_epoch(model: EnhancedNeuronLSTM, 
                       loader: DataLoader, 
                       criterion_class: nn.Module, 
                       optimizer: optim.Optimizer, 
                       device: torch.device, 
                       config: LSTMConfig, 
                       is_training: bool = True) -> Tuple[float, float]:
    """执行单个训练或验证轮次"""
    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    context_manager = torch.enable_grad() if is_training else torch.no_grad()
    with context_manager:
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            batch_size = batch_X.size(0)
            
            if is_training:
                optimizer.zero_grad()

            # 前向传播
            outputs, _, _ = model(batch_X) # 注意力权重现在是 None
            
            # 计算分类损失
            class_loss = criterion_class(outputs, batch_y)
            
            # 总损失现在就是分类损失
            combined_loss = class_loss 

            if is_training:
                combined_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                optimizer.step()

            # 统计
            total_loss += class_loss.item() * batch_size
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == batch_y).sum().item()
            total_samples += batch_size
            
    avg_loss = total_loss / total_samples
    accuracy = 100.0 * correct_predictions / total_samples
    
    return avg_loss, accuracy

def train_lstm_model(model: EnhancedNeuronLSTM,
                     train_loader: DataLoader,
                     val_loader: DataLoader,
                     criterion_class: nn.Module,
                     optimizer: optim.Optimizer,
                     scheduler: optim.lr_scheduler._LRScheduler,
                     device: torch.device,
                     config: LSTMConfig,
                     fold_num: int) -> Dict[str, List]:
    """单个 Fold 的模型训练流程"""
    print(f"--- [Fold {fold_num+1}/{config.k_folds}] 开始训练 --- ")
    start_time = datetime.datetime.now()

    best_val_loss = float('inf')
    best_val_acc = 0.0
    epochs_no_improve = 0
    best_model_state_acc = None # 只跟踪基于 Acc 的最佳状态，用于返回最佳 Acc

    metrics = {
        'train_losses': [], 'train_accuracies': [],
        'val_losses': [], 'val_accuracies': [], 'learning_rates': []
    }

    stopped_early = False
    final_epoch = 0

    for epoch in range(config.num_epochs):
        final_epoch = epoch + 1
        epoch_start_time = datetime.datetime.now()

        train_loss, train_acc = run_training_epoch(
            model, train_loader, criterion_class, optimizer, device, config, is_training=True
        )
        val_loss, val_acc = run_training_epoch(
            model, val_loader, criterion_class, optimizer, device, config, is_training=False
        )
        current_lr = optimizer.param_groups[0]['lr']
        epoch_duration = (datetime.datetime.now() - epoch_start_time).total_seconds()

        metrics['train_losses'].append(train_loss)
        metrics['train_accuracies'].append(train_acc)
        metrics['val_losses'].append(val_loss)
        metrics['val_accuracies'].append(val_acc)
        metrics['learning_rates'].append(current_lr)

        print(f"  [Fold {fold_num+1}] Epoch {epoch+1}/{config.num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f} | Time: {epoch_duration:.2f}s")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # 不需要保存模型状态，只记录最佳准确率
            print(f"    >> [Fold {fold_num+1}] 新的最佳验证准确率: {best_val_acc:.2f}%")

        if config.early_stopping_enabled and epochs_no_improve >= config.early_stopping_patience:
            print(f"  [Fold {fold_num+1}] 早停触发! (基于验证损失)")
            stopped_early = True
            break

    total_training_time = (datetime.datetime.now() - start_time).total_seconds()
    print(f"--- [Fold {fold_num+1}/{config.k_folds}] 训练结束 --- 总耗时: {total_training_time:.2f} 秒, "
          f"最佳验证准确率: {best_val_acc:.2f}% at Epoch {np.argmax(metrics['val_accuracies']) + 1 if metrics['val_accuracies'] else 'N/A'} --- ")

    metrics['best_val_acc_fold'] = best_val_acc # 记录该 fold 的最佳准确率
    return metrics

# --- 主执行函数 (重构为 K-Fold) ---
def main():
    """
    主训练流程 - 使用基于行为片段(原始)的 Stratified K-Fold (按时长分层),
    过滤跨边界序列, 并使用类别权重处理不平衡。
    """
    start_run_time = datetime.datetime.now()
    print(f"基于原始片段(时长分层, 边界过滤, 类别权重)的 K-Fold 训练脚本启动于: {start_run_time}") # Updated title
    config = None
    fold_accuracies = []

    try:
        # 1. 加载配置 & 初始化报告
        config = LSTMConfig()
        print("配置加载完成。")
        initialize_report(config) # 报告现在会包含 k_folds 参数

        # 2. 设置随机种子
        set_random_seed(config.random_seed)
        print(f"随机种子设置为: {config.random_seed}")

        # 3. 加载和预处理完整数据 (已过滤为核心类别)
        print("\n--- 加载、预处理并识别原始行为片段 --- ") # Updated title
        processor = NeuronDataProcessor(config)
        X, y = processor.preprocess_data() # X 未标准化, y 是核心类别标签
        num_classes = len(np.unique(y))
        input_size = X.shape[1]
        print(f"过滤后数据集大小: {X.shape}, 类别数: {num_classes}")

        # --- 识别原始行为片段 (不再分割) ---
        episodes = find_behavior_episodes(y, max_len=None) # Pass max_len=None
        print(f"识别得到 {len(episodes)} 个原始片段。") # Updated title
        if not episodes: raise ValueError("未能识别出任何片段。")
        # (可选) 打印片段统计信息
        from collections import Counter
        episode_labels = [label for _, _, label in episodes]
        print(f"原始片段类别分布: {Counter(episode_labels)}")

        # 4. 初始化基于原始片段的 K-Fold 分割 (按时长分层)
        folds_val_episodes = stratified_kfold_by_episode(episodes, config.k_folds, config.random_seed)
        print(f"\n--- 开始基于原始片段(时长分层)的 {config.k_folds}-Fold 交叉验证 --- ") # Updated title

        # 5. K-Fold 循环
        all_episode_indices = list(range(len(episodes))) # 用于查找训练片段

        for fold in range(config.k_folds):
            print(f"\n===== Fold {fold+1}/{config.k_folds} =====\n")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Device needed for weights

            # --- 获取当前 Fold 的训练/验证片段 ---
            val_episodes_fold = folds_val_episodes[fold]
            val_ep_set = set(val_episodes_fold)
            train_episodes_fold = [ep for ep in episodes if ep not in val_ep_set]

            # --- 从片段获取数据索引 ---
            train_indices = get_indices_from_episodes(train_episodes_fold)
            val_indices = get_indices_from_episodes(val_episodes_fold)

            # --- 数据准备 (当前 Fold) ---
            X_train_fold, X_val_fold = X[train_indices], X[val_indices]
            y_train_fold, y_val_fold = y[train_indices], y[val_indices]
            print(f"  训练集大小 (来自原始片段): {X_train_fold.shape}, 验证集大小 (来自原始片段): {X_val_fold.shape}")
            if len(X_train_fold) == 0 or len(X_val_fold) == 0:
                print(f"警告: Fold {fold+1} 训练集或验证集为空，跳过此折。检查片段识别逻辑。")
                fold_accuracies.append(0)
                continue

            # --- 标准化 (当前 Fold) ---
            scaler_fold = StandardScaler()
            X_train_scaled_fold = scaler_fold.fit_transform(X_train_fold)
            X_val_scaled_fold = scaler_fold.transform(X_val_fold)
            print("  数据标准化完成 (基于当前 Fold 训练集)")
            
            # --- 计算类别权重 (基于当前 Fold 训练集) ---
            unique_classes_fold = np.unique(y_train_fold)
            if len(unique_classes_fold) < num_classes:
                 print(f"警告: Fold {fold+1} 训练集类别数 ({len(unique_classes_fold)}) 少于总类别数 ({num_classes})。可能会影响权重计算或模型训练。")
                 # 可以在这里决定是跳过、使用默认权重还是继续
                 # 为了简单起见，我们继续，但 CrossEntropyLoss 可能会在遇到未见过的类时出错
                 # 更好的方法是确保 K-Fold 分割后每个 fold 至少包含每个类的一些样本

            class_weights_fold = compute_class_weight(
                'balanced', 
                classes=unique_classes_fold, # 使用实际存在的类别
                y=y_train_fold
            )
            # 将权重映射回完整的类别列表 (如果需要，但直接用在 Loss 里应该没问题只要 y_train_fold 标签正确)
            # 创建权重张量并移到设备
            # 注意：compute_class_weight 返回与 unique_classes_fold 顺序对应的权重
            # CrossEntropyLoss 的 weight 参数期望一个长度为 C 的张量，其索引对应类别标签 0 到 C-1
            # 我们需要创建一个完整的权重张量
            full_class_weights = torch.ones(num_classes, dtype=torch.float) # 默认权重为 1
            for i, class_label in enumerate(unique_classes_fold):
                if class_label < num_classes: # 确保类别标签在范围内
                    full_class_weights[class_label] = class_weights_fold[i]
                else:
                     print(f"警告: Fold {fold+1} 训练集中发现无效类别标签 {class_label}，忽略其权重。")
            
            class_weights_tensor = full_class_weights.to(device)
            print(f"  计算得到的类别权重 (Fold {fold+1}): {class_weights_tensor.cpu().numpy().round(2)}")
            # ------------------------------------------

            # --- 数据集创建 (初步) ---
            train_dataset_fold = NeuronDataset(X_train_scaled_fold, y_train_fold, config.sequence_length)
            val_dataset_fold = NeuronDataset(X_val_scaled_fold, y_val_fold, config.sequence_length)
            
            # --- 过滤跨边界序列 --- 
            # ... (过滤逻辑保持不变) ...
            num_train_before_filter = len(train_dataset_fold)
            valid_train_indices_for_subset = []
            for j in range(num_train_before_filter):
                 seq_end_idx_in_subset = j + config.sequence_length - 1
                 if seq_end_idx_in_subset < len(train_indices):
                     original_start_idx = train_indices[j]
                     is_contiguous = True
                     for k in range(1, config.sequence_length):
                         if j+k >= len(train_indices) or train_indices[j+k] != original_start_idx + k:
                             is_contiguous = False
                             break
                     if is_contiguous:
                         valid_train_indices_for_subset.append(j)
            filtered_train_dataset = Subset(train_dataset_fold, valid_train_indices_for_subset)
            num_train_after_filter = len(filtered_train_dataset)
            print(f"  训练集序列过滤: {num_train_before_filter} -> {num_train_after_filter}")

            num_val_before_filter = len(val_dataset_fold)
            valid_val_indices_for_subset = []
            for j in range(num_val_before_filter):
                 seq_end_idx_in_subset = j + config.sequence_length - 1
                 if seq_end_idx_in_subset < len(val_indices):
                     original_start_idx = val_indices[j]
                     is_contiguous = True
                     for k in range(1, config.sequence_length):
                         if j+k >= len(val_indices) or val_indices[j+k] != original_start_idx + k:
                             is_contiguous = False
                             break
                     if is_contiguous:
                         valid_val_indices_for_subset.append(j)
            filtered_val_dataset = Subset(val_dataset_fold, valid_val_indices_for_subset)
            num_val_after_filter = len(filtered_val_dataset)
            print(f"  验证集序列过滤: {num_val_before_filter} -> {num_val_after_filter}")

            if num_train_after_filter == 0 or num_val_after_filter == 0:
                print(f"警告: Fold {fold+1} 过滤后训练或验证序列数量为零，跳过此折。")
                fold_accuracies.append(0)
                continue

            # --- 数据加载器 (使用过滤后的数据集) ---
            train_loader_fold = DataLoader(filtered_train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
            val_loader_fold = DataLoader(filtered_val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)
            print(f"  数据加载器创建完成 (过滤后): 训练集 {len(train_loader_fold)} 批次, 验证集 {len(val_loader_fold)} 批次")

            # --- 初始化模型、损失(带权重)、优化器、调度器 ---
            model_fold = EnhancedNeuronLSTM(
                input_size=input_size, hidden_size=config.hidden_size, num_layers=config.num_layers,
                num_classes=num_classes, dropout=config.dropout
            ).to(device)
            # 使用带权重的损失函数
            criterion_class_fold = nn.CrossEntropyLoss(weight=class_weights_tensor) 
            optimizer_fold = optim.AdamW(model_fold.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
            scheduler_fold = optim.lr_scheduler.ReduceLROnPlateau(optimizer_fold, mode='min', factor=0.5, patience=config.lr_scheduler_patience, verbose=False) # 使用 config 中的 patience
            print(f"  模型 (Fold {fold+1})、带权重损失、优化器、调度器初始化完成。使用设备: {device}")

            # --- 训练当前 Fold ---
            metrics_fold = train_lstm_model(
                model_fold, train_loader_fold, val_loader_fold,
                criterion_class_fold, optimizer_fold, scheduler_fold,
                device, config, fold
            )
            fold_accuracies.append(metrics_fold.get('best_val_acc_fold', 0))

        # 6. 计算并打印总体交叉验证结果
        print("\n===== 基于原始片段(时长分层, 边界过滤, 类别权重)的 K-Fold 交叉验证完成 =====") # Updated title
        valid_fold_accuracies = [acc for acc in fold_accuracies if acc > 0]
        if valid_fold_accuracies:
            mean_accuracy = np.mean(valid_fold_accuracies)
            std_accuracy = np.std(valid_fold_accuracies)
            print(f"平均最佳验证准确率 ({len(valid_fold_accuracies)}/{config.k_folds} 折): {mean_accuracy:.2f}% (+/- {std_accuracy:.2f}%)")
            print(f"各折准确率: {[f'{acc:.2f}%' for acc in fold_accuracies]}")
        else:
            print("错误：所有 Fold 都未能成功完成训练。")
            mean_accuracy = 0
            std_accuracy = 0

        # --- 追加 K-Fold 结果到报告 ---
        run_duration_kfold = (datetime.datetime.now() - start_run_time).total_seconds() # 获取总秒数
        append_kfold_summary(config, fold_accuracies, run_duration_kfold)
        # ---------------------------------

        # --- 7. 训练最终模型 (基于完整训练集和当前最佳配置) ---
        print("\n===== 开始训练最终模型 (使用完整训练集) =====")
        final_train_start_time = datetime.datetime.now()
        
        # --- 准备完整训练数据 (不包含测试集) ---
        # 我们需要原始未缩放的 X 和 y 来重新分割
        # 注意: `split_data` 会使用 config 中的比例，我们需要确保它返回的是训练+验证部分
        # 我们假设 split_data 返回 (X_train, y_train, X_val, y_val, X_test, y_test)
        # 我们需要合并 X_train 和 X_val 来形成完整的训练集
        X_train_orig, y_train_orig, X_val_orig, y_val_orig, _, _ = split_data(X, y, config)
        X_train_full = np.concatenate((X_train_orig, X_val_orig), axis=0)
        y_train_full = np.concatenate((y_train_orig, y_val_orig), axis=0)
        print(f"  完整训练集大小: {X_train_full.shape}")

        # --- 训练最终 Scaler 并保存 ---
        final_scaler = StandardScaler()
        X_train_full_scaled = final_scaler.fit_transform(X_train_full)
        save_scaler(final_scaler, config.scaler_path)
        print(f"  在完整训练集上训练的最终 Scaler 已保存到: {config.scaler_path}")

        # --- 创建最终训练数据集和加载器 (应用边界过滤) ---
        # 需要完整的原始索引来做边界过滤
        # 这里假设 y_train_full 对应的是合并后数据的标签顺序
        # 我们需要找到合并后数据的原始片段信息
        # 最简单的方法是重新在 y_train_full 上运行 find_behavior_episodes
        # 注意：这假设合并后的 y_train_full 仍然保持了时间顺序
        train_full_episodes = find_behavior_episodes(y_train_full, max_len=None) # 获取合并后数据的片段
        train_full_indices = get_indices_from_episodes(train_full_episodes) # 获取片段对应的索引 (相对于 y_train_full)
        
        final_train_dataset = NeuronDataset(X_train_full_scaled, y_train_full, config.sequence_length)
        num_final_train_before_filter = len(final_train_dataset)
        valid_final_train_indices = []
        for j in range(num_final_train_before_filter):
             seq_end_idx_in_subset = j + config.sequence_length - 1
             if seq_end_idx_in_subset < len(train_full_indices):
                 original_start_idx = train_full_indices[j]
                 is_contiguous = True
                 for k in range(1, config.sequence_length):
                     if j+k >= len(train_full_indices) or train_full_indices[j+k] != original_start_idx + k:
                         is_contiguous = False
                         break
                 if is_contiguous:
                     valid_final_train_indices.append(j)
        
        filtered_final_train_dataset = Subset(final_train_dataset, valid_final_train_indices)
        num_final_train_after_filter = len(filtered_final_train_dataset)
        print(f"  最终训练集序列过滤: {num_final_train_before_filter} -> {num_final_train_after_filter}")
        
        if num_final_train_after_filter == 0:
            print("错误: 最终训练集过滤后序列数量为零，无法训练最终模型。")
        else:
            final_train_loader = DataLoader(filtered_final_train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
            print(f"  最终训练数据加载器创建完成: {len(final_train_loader)} 批次")

            # --- 初始化最终模型、损失、优化器 ---
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            final_model = EnhancedNeuronLSTM(
                input_size=input_size, hidden_size=config.hidden_size, num_layers=config.num_layers,
                num_classes=num_classes, dropout=config.dropout
            ).to(device)
            
            # 重新计算类别权重 (基于完整训练集)
            unique_classes_full = np.unique(y_train_full)
            class_weights_full = compute_class_weight('balanced', classes=unique_classes_full, y=y_train_full)
            full_class_weights_final = torch.ones(num_classes, dtype=torch.float)
            for i, class_label in enumerate(unique_classes_full):
                if class_label < num_classes: full_class_weights_final[class_label] = class_weights_full[i]
            class_weights_tensor_final = full_class_weights_final.to(device)
            print(f"  计算得到的最终类别权重: {class_weights_tensor_final.cpu().numpy().round(2)}")
            
            criterion_final = nn.CrossEntropyLoss(weight=class_weights_tensor_final)
            optimizer_final = optim.AdamW(final_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
            # 注意: 不再需要 scheduler 或早停，因为我们训练固定轮数
            print(f"  最终模型、带权重损失、优化器初始化完成。使用设备: {device}")
            
            # --- 训练固定轮数 (例如 20 轮) ---
            final_train_epochs = 20 #! 可以调整或基于 K-Fold 结果动态确定
            print(f"--- 开始训练最终模型 {final_train_epochs} 轮 ---")
            for epoch in range(1, final_train_epochs + 1):
                epoch_start_time = datetime.datetime.now()
                # 只进行训练轮次
                train_loss, train_acc = run_training_epoch(
                    final_model, final_train_loader, criterion_final, optimizer_final, 
                    device, config, is_training=True
                )
                epoch_duration = (datetime.datetime.now() - epoch_start_time).total_seconds()
                print(f"  [Final Train] Epoch {epoch}/{final_train_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Time: {epoch_duration:.2f}s")
            
            # --- 保存最终模型 ---
            final_model_save_path = config.model_path
            try:
                torch.save({
                    'epoch': final_train_epochs, 
                    'model_state_dict': final_model.state_dict(),
                    'optimizer_state_dict': optimizer_final.state_dict(),
                    # 可以选择保存 config 对象或关键参数
                    'config_params': vars(config) 
                }, final_model_save_path)
                print(f"--- 最终模型已保存到: {final_model_save_path} ---")
            except Exception as e:
                print(f"保存最终模型时出错: {e}")
        
        final_train_duration = (datetime.datetime.now() - final_train_start_time).total_seconds()
        print(f"===== 最终模型训练结束 ===== 总耗时: {final_train_duration:.2f} 秒")
        # ------------------------------------------------------------

        # --- 清理/收尾 ---
        total_run_duration = (datetime.datetime.now() - start_run_time).total_seconds()
        print(f"\nK-Fold 及最终模型训练完成于: {datetime.datetime.now()}, 总耗时: {total_run_duration:.2f} 秒") # Updated title
        print(f"最终模型、Scaler 及 K-Fold 摘要已记录到报告文件。请运行评估脚本进行测试集评估。")

    except Exception as e:
        print(f"发生未知错误: {e}")
        if config:
            with open(config.error_log, 'a') as f:
                import traceback
                traceback.print_exc(file=f)
                f.write(f"\n训练脚本 (含最终模型训练) 在 {datetime.datetime.now()} 异常终止。\n") # Updated title
        else: print("配置未加载，无法写入错误日志。")

if __name__ == "__main__":
    main() 