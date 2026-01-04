import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
import argparse
import random
from typing import Optional
from sklearn.metrics import classification_report
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from model import ImprovedGCN
from process import load_data, oversample_data, compute_correlation_matrix, create_pyg_dataset, visualize_graph, enhance_balanced_dataset
from train import train_model, evaluate_model, plot_confusion_matrix, plot_training_metrics, plot_learning_curve

def setup_result_directory(input_file_path, min_samples=None, effect_threshold=None, effect_filter_mode=None):
    # 提取文件名（不含路径和扩展名）
    file_name = os.path.basename(input_file_path)
    file_name = os.path.splitext(file_name)[0]
    
    # 添加最小样本数信息（如果提供）
    if min_samples is not None:
        file_name = f"{file_name}"
    
    # 添加时间戳以确保唯一性
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = ""
    if effect_threshold is not None:
        mode = str(effect_filter_mode or 'gt').lower()
        if mode == 'lt':
            suffix = f"_阈值{effect_threshold}_反向"
        else:
            suffix = f"_阈值{effect_threshold}"
    result_dir = f"../results/{file_name}_{timestamp}{suffix}"
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    return result_dir

def setup_matplotlib_fonts():
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Arial']  # 优先使用中文黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    # 设置坐标轴标签字体大小和粗细 - 进一步增大字体
    plt.rcParams['axes.labelsize'] = 24  # 坐标轴标签字体大小
    plt.rcParams['axes.labelweight'] = 'bold'  # 坐标轴标签字体粗细
    plt.rcParams['xtick.labelsize'] = 20  # X轴刻度字体大小
    plt.rcParams['ytick.labelsize'] = 20  # Y轴刻度字体大小
    plt.rcParams['axes.titlesize'] = 28  # 图表标题字体大小
    plt.rcParams['axes.titleweight'] = 'bold'  # 图表标题字体粗细
    plt.rcParams['legend.fontsize'] = 20  # 图例字体大小
    
    # 设置线条样式
    plt.rcParams['lines.linewidth'] = 3  # 默认线条粗细
    plt.rcParams['axes.linewidth'] = 2  # 坐标轴边框粗细
    plt.rcParams['grid.linewidth'] = 1  # 网格线粗细
    
    # 检查字体是否正确设置
    # print("可用字体:", mpl.font_manager.findSystemFonts(fontpaths=None, fontext="ttf"))
    # 测试中文显示
    fig, ax = plt.figure(), plt.axes()
    ax.set_title('测试中文显示')
    plt.close(fig)

def set_seed(seed: int):
    seed = int(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed

def run_single_experiment(
    data_file: str,
    position_file: str,
    min_samples: int,
    effect_size_file: str,
    effect_threshold: float,
    effect_filter_mode: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    result_dir: str,
    init_model_path: Optional[str] = None,
    gpu_id: int = -1,
):
    seed = set_seed(seed)
    if torch.cuda.is_available():
        if int(gpu_id) >= 0:
            device = torch.device(f'cuda:{int(gpu_id)}')
            torch.cuda.set_device(device)
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    with open(f'{result_dir}/config.txt', 'w', encoding='utf-8') as f:
        f.write(f"数据文件: {data_file}\n")
        f.write(f"神经元位置文件: {position_file}\n")
        f.write(f"最小样本数: {min_samples}\n")
        f.write(f"效应量文件: {effect_size_file}\n")
        f.write(f"效应量阈值: {effect_threshold}\n")
        f.write(f"效应量筛选模式: {effect_filter_mode}\n")
        f.write(f"训练集比例: {train_ratio}\n")
        f.write(f"验证集比例: {val_ratio}\n")
        f.write(f"测试集比例: {1 - train_ratio - val_ratio}\n")
        f.write(f"gpu_id: {gpu_id}\n")
        f.write(f"设备: {device}\n")
        f.write(f"随机种子: {seed}\n")
        f.write(f"初始化模型: {init_model_path or 'None'}\n")
        f.write(f"训练开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 设定最小样本数为50，将样本数少于50的标签过滤
    features, labels, class_weights, class_names = load_data(
        data_file,
        min_samples=min_samples,
        effect_size_path=effect_size_file,
        effect_threshold=effect_threshold,
        effect_filter_mode=effect_filter_mode
    )
    
    # 使用增强的数据平衡方法处理不平衡数据
    # 选择合适的数据增强策略
    augmentation_strategy = 'comprehensive'  # 可选: 'basic', 'gan', 'vae', 'comprehensive', 'adasyn', 'none'
    
    if augmentation_strategy == 'basic':
        # 基础策略: 组合重采样 + 时间序列增强
        features_resampled, labels_resampled = enhance_balanced_dataset(
            features, labels, 
            methods=['combined', 'timeseries'],
            random_state=seed
        )
    elif augmentation_strategy == 'gan':
        # GAN策略: 先下采样再用GAN生成
        features_resampled, labels_resampled = enhance_balanced_dataset(
            features, labels, 
            methods=['random_under', 'gan'],
            random_state=seed
        )
    elif augmentation_strategy == 'vae':
        # VAE策略: 先下采样再用VAE生成
        features_resampled, labels_resampled = enhance_balanced_dataset(
            features, labels, 
            methods=['random_under', 'vae'],
            random_state=seed
        )
    elif augmentation_strategy == 'comprehensive':
        # 综合策略: 先下采样再组合多种生成方法
        features_resampled, labels_resampled = enhance_balanced_dataset(
            features, labels, 
            methods=['combined', 'timeseries', 'gan', 'vae'],
            random_state=seed
        )
    elif augmentation_strategy == 'adasyn':
        # ADASYN策略: 使用ADASYN自适应合成采样
        features_resampled, labels_resampled = enhance_balanced_dataset(
            features, labels,
            methods=['adasyn'],
            random_state=seed
        )
    elif augmentation_strategy == 'none':
        # 不使用任何数据增强方法，直接使用原始数据
        print("不使用数据增强方法，保持原始数据分布")
        features_resampled, labels_resampled = features, labels
    else:
        # 默认使用SMOTE
        features_resampled, labels_resampled = oversample_data(
            features, labels, ramdom_state=seed, method='smote'
        )
    print(f"最终重采样后特征形状: {features_resampled.shape}")

    # 划分训练集和验证集和测试集 (60%/20%/20%)
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError(f"非法划分比例: train_ratio={train_ratio}, val_ratio={val_ratio}")

    X_train, X_temp, y_train, y_temp = train_test_split(
        features_resampled, labels_resampled,
        test_size=(1 - train_ratio),
        random_state=seed,
        stratify=labels_resampled
    )
    
    val_size = val_ratio / (1 - train_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_size),
        random_state=seed,
        stratify=y_temp
    )

    correlation_matrix = compute_correlation_matrix(X_train)
    print(f"Correlation matrix shape: {correlation_matrix.shape}")

    # 可视化相关性矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, cmap='coolwarm', center=0)
    plt.title("Neuron Correlation Matrix", fontsize=28, fontweight='bold')
    plt.xlabel("Neuron Index", fontsize=24, fontweight='bold')
    plt.ylabel("Neuron Index", fontsize=24, fontweight='bold')
    plt.xticks(fontsize=20, fontweight='bold')
    plt.yticks(fontsize=20, fontweight='bold')
    plt.savefig(f'{result_dir}/correlation_matrix.png')
    plt.close()
    
    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")
    # 生成图数据
    train_data_list = create_pyg_dataset(X_train, y_train, correlation_matrix)
    val_data_list = create_pyg_dataset(X_val, y_val, correlation_matrix)
    test_data_list = create_pyg_dataset(X_test, y_test, correlation_matrix)

    # 使用真实空间位置可视化神经元图
    visualize_graph(
        train_data_list, 
        sample_index=0, 
        title="Neuron Network Topology (Training Samples)", 
        result_dir=result_dir,
        position_file=position_file,
        seed=seed
    )

    # 统计一些图结构特征
    num_nodes = train_data_list[0].x.shape[0]  # 节点数量
    avg_edges = sum(data.edge_index.shape[1] for data in train_data_list) / len(train_data_list) / 2  # 平均边数量
    print(f"每个图的节点数量: {num_nodes}, 平均边数量: {avg_edges}")
    
    # 创建数据加载器
    train_loader = DataLoader(train_data_list, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data_list, batch_size=32)
    test_loader = DataLoader(test_data_list, batch_size=32)

    model = ImprovedGCN(
        num_features=1,
        hidden_dim=64,
        num_classes=len(np.unique(labels)),
        dropout=0.3
    ).to(device)

    if init_model_path:
        init_model_path = str(init_model_path)
        if os.path.exists(init_model_path):
            state_dict = torch.load(init_model_path, map_location=device)
            model.load_state_dict(state_dict, strict=True)
            print(f"已从上一轮权重初始化模型: {init_model_path}")
        else:
            raise FileNotFoundError(f"初始化模型文件不存在: {init_model_path}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数数量: {total_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )
    
    # 训练模型
    best_val_f1 = 0
    best_epoch = 0
    # patience = 20  # early stopping patience
    # epochs_no_improve = 0
    max_epochs = 300
    
    history = {
        'train': {
            'loss': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        },
        'val': {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
    }
    
    print("开始训练...")
    for epoch in range(max_epochs):
        train_metrics = train_model(model, train_loader, optimizer, device, class_weights)
        # 验证
        val_metrics = evaluate_model(model, val_loader, device)
        # 更新学习率
        scheduler.step(val_metrics['f1'])
        # 记录指标
        for metric in train_metrics:
            history['train'][metric].append(train_metrics[metric])
        for metric in val_metrics:
            if metric not in ['predictions', 'labels']:
                history['val'][metric].append(val_metrics[metric])
        
        # 保存最佳模型（基于验证集F1分数）
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_epoch = epoch
            torch.save(model.state_dict(), f'{result_dir}/best_model.pth')
            print(f"✓ Epoch {epoch+1}: 保存新的最佳模型，验证F1分数: {best_val_f1:.4f}")
        # 打印进度
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch {epoch+1}/{max_epochs}:')
            print(f'  Train - Loss: {train_metrics["loss"]:.4f}, Accuracy: {train_metrics["accuracy"]:.4f}, F1: {train_metrics["f1"]:.4f}')
            print(f'  Val   - Accuracy: {val_metrics["accuracy"]:.4f}, F1: {val_metrics["f1"]:.4f}')
        # 继续训练直到最大epoch数
    print(f"\n训练完成! 最佳验证F1分数: {best_val_f1:.4f} (epoch {best_epoch+1})")
    torch.save(model.state_dict(), f'{result_dir}/last_model.pth')
    # 绘制训练指标
    plot_training_metrics(history['train'], history['val'], result_dir=result_dir)
    plot_learning_curve(history['train'], history['val'], result_dir=result_dir)
    print("加载最佳模型进行最终测试...")
    best_model_path = f'{result_dir}/best_model.pth'
    if not os.path.exists(best_model_path):
        torch.save(model.state_dict(), best_model_path)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_metrics = evaluate_model(model, test_loader, device)
    
    print("\n测试集分类报告:")
    print(classification_report(
        test_metrics['labels'], 
        test_metrics['predictions'], 
        target_names=class_names
    ))
    
    # 绘制混淆矩阵
    plot_confusion_matrix(test_metrics['labels'], test_metrics['predictions'], class_names, result_dir=result_dir)
    
    # 保存最终测试结果
    with open(f'{result_dir}/test_results.txt', 'w', encoding='utf-8') as f:
        f.write(f"最小样本数阈值: {min_samples}\n")
        f.write(f"效应量阈值: {effect_threshold}\n")
        f.write(f"剩余标签数量: {len(class_names)}\n\n")
        f.write(f"测试集准确率: {test_metrics['accuracy']:.4f}\n")
        f.write(f"测试集精确率: {test_metrics['precision']:.4f}\n")
        f.write(f"测试集召回率: {test_metrics['recall']:.4f}\n")
        f.write(f"测试集F1分数: {test_metrics['f1']:.4f}\n\n")
        f.write("分类报告:\n")
        f.write(classification_report(
            test_metrics['labels'], 
            test_metrics['predictions'], 
            target_names=class_names
        ))
    
    print(f"\n最终测试结果:")
    print(f"  最小样本数阈值: {min_samples}")
    print(f"  剩余标签数量: {len(class_names)}")
    print(f"  准确率: {test_metrics['accuracy']:.4f}")
    print(f"  精确率: {test_metrics['precision']:.4f}")
    print(f"  召回率: {test_metrics['recall']:.4f}")
    print(f"  F1分数: {test_metrics['f1']:.4f}")

    return {
        'accuracy': float(test_metrics['accuracy']),
        'precision': float(test_metrics['precision']),
        'recall': float(test_metrics['recall']),
        'f1': float(test_metrics['f1']),
        'best_val_f1': float(best_val_f1),
        'best_epoch': int(best_epoch + 1),
        'num_classes': int(len(class_names)),
        'num_samples_total': int(len(labels_resampled)),
        'num_samples_train': int(len(y_train)),
        'num_samples_val': int(len(y_val)),
        'num_samples_test': int(len(y_test)),
    }

def main():
    setup_matplotlib_fonts()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default='../datasets/no.29800930openfield_CellVideo0_corrected_0_cell_trace.xlsx')
    parser.add_argument('--position_file', type=str, default='../datasets/no.29800930openfield神经元编号位置图.csv')
    parser.add_argument('--min_samples', type=int, default=50)
    parser.add_argument('--effect_size_file', type=str, default='../datasets/effect_sizes_no.29800930openfield_CellVideo0_corrected_0_cell_trace.csv')
    parser.add_argument('--effect_threshold', type=float, default=0.3)
    parser.add_argument('--effect_filter_mode', type=str, default='gt', choices=['gt', 'lt'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--train_ratio', type=float, default=0.6)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--gpu_id', type=int, default=-1)
    parser.add_argument(
        '--run_init',
        type=str,
        default='fresh',
        choices=['fresh', 'prev_best', 'prev_last'],
    )
    args = parser.parse_args()

    data_file = args.data_file
    position_file = args.position_file
    min_samples = args.min_samples
    effect_size_file = args.effect_size_file
    effect_threshold = args.effect_threshold
    effect_filter_mode = args.effect_filter_mode

    result_dir = setup_result_directory(
        data_file,
        min_samples=min_samples,
        effect_threshold=effect_threshold,
        effect_filter_mode=effect_filter_mode
    )
    print(f"结果将保存到: {result_dir}")

    if args.num_runs > 1:
        all_results = []
        initial_seed = int(args.seed)
        prev_run_best_model_path = None
        prev_run_last_model_path = None

        for run_idx in range(int(args.num_runs)):
            current_seed = initial_seed + run_idx
            run_dir = os.path.join(result_dir, f"run_{run_idx + 1:02d}_seed{current_seed}")
            os.makedirs(run_dir, exist_ok=True)

            print("\n" + "=" * 80)
            print(f"开始第 {run_idx + 1}/{args.num_runs} 次训练 (Seed = {current_seed})")
            print("=" * 80)

            init_model_path = None
            if str(args.run_init) == 'prev_best' and prev_run_best_model_path:
                init_model_path = prev_run_best_model_path
            elif str(args.run_init) == 'prev_last' and prev_run_last_model_path:
                init_model_path = prev_run_last_model_path

            metrics = run_single_experiment(
                data_file=data_file,
                position_file=position_file,
                min_samples=min_samples,
                effect_size_file=effect_size_file,
                effect_threshold=effect_threshold,
                effect_filter_mode=effect_filter_mode,
                seed=current_seed,
                train_ratio=float(args.train_ratio),
                val_ratio=float(args.val_ratio),
                result_dir=run_dir,
                init_model_path=init_model_path,
                gpu_id=int(args.gpu_id),
            )
            metrics.update({'run': int(run_idx + 1), 'seed': int(current_seed)})
            all_results.append(metrics)
            prev_run_best_model_path = os.path.join(run_dir, "best_model.pth")
            prev_run_last_model_path = os.path.join(run_dir, "last_model.pth")

        results_df = pd.DataFrame(all_results).sort_values(by='run')
        results_df.to_csv(f'{result_dir}/multi_run_results.csv', index=False, encoding='utf-8-sig')

        metric_cols = [c for c in ['accuracy', 'precision', 'recall', 'f1'] if c in results_df.columns]
        summary = results_df[metric_cols].agg(['mean', 'std']).to_dict()
        with open(f'{result_dir}/multi_run_summary.txt', 'w', encoding='utf-8') as f:
            f.write(f"num_runs: {args.num_runs}\n")
            f.write(f"initial_seed: {initial_seed}\n")
            f.write(f"train_ratio: {args.train_ratio}\n")
            f.write(f"val_ratio: {args.val_ratio}\n")
            f.write(f"test_ratio: {1 - args.train_ratio - args.val_ratio}\n")
            f.write(f"run_init: {args.run_init}\n")
            for m in metric_cols:
                m_mean = summary[m]['mean']
                m_std = summary[m]['std']
                f.write(f"{m}: mean={m_mean:.6f}, std={m_std:.6f}\n")
        print(f"\n多次运行汇总已保存: {result_dir}/multi_run_results.csv 与 {result_dir}/multi_run_summary.txt")
        return

    run_single_experiment(
        data_file=data_file,
        position_file=position_file,
        min_samples=min_samples,
        effect_size_file=effect_size_file,
        effect_threshold=effect_threshold,
        effect_filter_mode=effect_filter_mode,
        seed=int(args.seed),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        result_dir=result_dir,
        init_model_path=None,
        gpu_id=int(args.gpu_id),
    )

if __name__ == "__main__":
    main()
