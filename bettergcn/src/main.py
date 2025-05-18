import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
import matplotlib as mpl
from sklearn.metrics import classification_report
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from model import ImprovedGCN
from process import load_data, oversample_data, compute_correlation_matrix, create_pyg_dataset, visualize_graph, enhance_balanced_dataset
from train import train_model, evaluate_model, plot_confusion_matrix, plot_training_metrics, plot_learning_curve

def setup_result_directory(input_file_path, min_samples=None):
    # 提取文件名（不含路径和扩展名）
    file_name = os.path.basename(input_file_path)
    file_name = os.path.splitext(file_name)[0]
    
    # 添加最小样本数信息（如果提供）
    if min_samples is not None:
        file_name = f"{file_name}"
    
    # 添加时间戳以确保唯一性
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = f"../result/{file_name}"
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    return result_dir

def setup_matplotlib_fonts():
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Arial']  # 优先使用中文黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    # 检查字体是否正确设置
    # print("可用字体:", mpl.font_manager.findSystemFonts(fontpaths=None, fontext="ttf"))
    # 测试中文显示
    fig, ax = plt.figure(), plt.axes()
    ax.set_title('测试中文显示')
    plt.close(fig)


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    setup_matplotlib_fonts()
    
    # 定义数据文件路径和最小样本数
    data_file = '../datasets/EMtrace01.xlsx'
    position_file = '../datasets/EMtrace01_Max_position.csv'
    min_samples = 50
    
    # 使用数据文件名和最小样本数来设置结果目录
    result_dir = setup_result_directory(data_file, min_samples)
    print(f"结果将保存到: {result_dir}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 保存实验配置
    with open(f'{result_dir}/config.txt', 'w', encoding='utf-8') as f:
        f.write(f"数据文件: {data_file}\n")
        f.write(f"神经元位置文件: {position_file}\n")
        f.write(f"最小样本数: {min_samples}\n")
        f.write(f"设备: {device}\n")
        f.write(f"随机种子: 42\n")
        f.write(f"训练开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 设定最小样本数为50，将样本数少于50的标签过滤
    features, labels, class_weights, class_names = load_data(data_file, min_samples=min_samples)
    
    # 使用增强的数据平衡方法处理不平衡数据
    # 选择合适的数据增强策略
    augmentation_strategy = 'comprehensive'  # 可选: 'basic', 'gan', 'vae', 'comprehensive'
    
    if augmentation_strategy == 'basic':
        # 基础策略: 组合重采样 + 时间序列增强
        features_resampled, labels_resampled = enhance_balanced_dataset(
            features, labels, 
            methods=['combined', 'timeseries']
        )
    elif augmentation_strategy == 'gan':
        # GAN策略: 先下采样再用GAN生成
        features_resampled, labels_resampled = enhance_balanced_dataset(
            features, labels, 
            methods=['random_under', 'gan']
        )
    elif augmentation_strategy == 'vae':
        # VAE策略: 先下采样再用VAE生成
        features_resampled, labels_resampled = enhance_balanced_dataset(
            features, labels, 
            methods=['random_under', 'vae']
        )
    elif augmentation_strategy == 'comprehensive':
        # 综合策略: 先下采样再组合多种生成方法
        features_resampled, labels_resampled = enhance_balanced_dataset(
            features, labels, 
            methods=['combined', 'timeseries', 'gan', 'vae']
        )
    else:
        # 默认使用SMOTE
        features_resampled, labels_resampled = oversample_data(
            features, labels, ramdom_state=42, method='smote'
        )
    
    print(f"最终重采样后特征形状: {features_resampled.shape}")

    # 划分训练集和验证集和测试集 (60%/20%/20%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        features_resampled, labels_resampled, test_size=0.4, 
        random_state=42, stratify=labels_resampled
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, 
        random_state=42, stratify=y_temp
    )

    correlation_matrix = compute_correlation_matrix(X_train)
    print(f"Correlation matrix shape: {correlation_matrix.shape}")

    # 可视化相关性矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, cmap='coolwarm', center=0)
    plt.title("Neuron Correlation Matrix")
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
        position_file=position_file
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


    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数数量: {total_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )
    
    # 训练模型
    best_val_f1 = 0
    best_epoch = 0
    patience = 20  # early stopping patience
    epochs_no_improve = 0
    max_epochs = 250
    
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
            epochs_no_improve = 0
            torch.save(model.state_dict(), f'{result_dir}/best_model.pth')
            print(f"✓ Epoch {epoch+1}: 保存新的最佳模型，验证F1分数: {best_val_f1:.4f}")
        else:
            epochs_no_improve += 1
        # 打印进度
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch {epoch+1}/{max_epochs}:')
            print(f'  Train - Loss: {train_metrics["loss"]:.4f}, Accuracy: {train_metrics["accuracy"]:.4f}, F1: {train_metrics["f1"]:.4f}')
            print(f'  Val   - Accuracy: {val_metrics["accuracy"]:.4f}, F1: {val_metrics["f1"]:.4f}')
        # 早停检查
        if epochs_no_improve >= patience:
            print(f'\n早停! {patience}个epoch没有提高。')
            print(f'最佳F1分数: {best_val_f1:.4f} (epoch {best_epoch+1})')
            break
    print(f"\n训练完成! 最佳验证F1分数: {best_val_f1:.4f} (epoch {best_epoch+1})")
    # 绘制训练指标
    plot_training_metrics(history['train'], history['val'], result_dir=result_dir)
    plot_learning_curve(history['train'], history['val'], result_dir=result_dir)
    print("加载最佳模型进行最终测试...")
    model.load_state_dict(torch.load(f'{result_dir}/best_model.pth'))
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

if __name__ == "__main__":
    main()