import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
import time
import json
import argparse
import matplotlib as mpl
from sklearn.metrics import classification_report
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from model import ImprovedGCN, PureGCN, PureGAT #, PureGraphSAGE
from process import load_data, oversample_data # , compute_correlation_matrix, create_pyg_dataset, visualize_graph
from train import train_model, evaluate_model, plot_confusion_matrix, plot_training_metrics, plot_learning_curve

# 模型字典，方便通过字符串参数选择模型
MODEL_DICT = {
    'hybrid': ImprovedGCN,
    'gcn': PureGCN,
    'gat': PureGAT,
    # sage': PureGraphSAGE
}

def setup_result_directory(model_name):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"result/{model_name}"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    result_dir = f"{base_dir}/{timestamp}"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    return result_dir

def setup_matplotlib_fonts():
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.figure(), plt.axes()
    ax.set_title('测试中文显示')
    plt.close(fig)

def run_single_experiment(model_name, seed, data_path='../../data', hidden_dim=64, dropout=0.3, save_all_results=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 如果保存所有实验结果，则为每次实验创建单独目录
    if save_all_results:
        result_dir = setup_result_directory(model_name)
    else:
        # 如果不保存所有结果，只创建模型根目录
        result_dir = f"result/{model_name}"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 记录开始时间
    begin_time = time.time()
    
    # 检查数据路径类型并加载数据
    if data_path.endswith('.csv'):
        # 原有的CSV数据处理流程
        features, labels, class_weights, class_names = load_data(data_path)
        features_resampled, labels_resampled = oversample_data(features, labels, ramdom_state=seed)
        
        X_train, X_temp, y_train, y_temp = train_test_split(
            features_resampled, labels_resampled, test_size=0.4, 
            random_state=seed, stratify=labels_resampled
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, 
            random_state=seed, stratify=y_temp
        )
        
        correlation_matrix = compute_correlation_matrix(X_train)
        
        if save_all_results:
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, cmap='coolwarm', center=0)
            plt.title("神经元相关性矩阵")
            plt.savefig(f'{result_dir}/correlation_matrix.png')
            plt.close()
        
        train_data_list = create_pyg_dataset(X_train, y_train, correlation_matrix)
        val_data_list = create_pyg_dataset(X_val, y_val, correlation_matrix)
        test_data_list = create_pyg_dataset(X_test, y_test, correlation_matrix)
        
        if save_all_results:
            visualize_graph(train_data_list, sample_index=0, title="训练样本神经元图", result_dir=result_dir)
    else:
        # 新的图数据处理流程
        data_list, class_weights, class_names = load_data(data_path)
        
        if not data_list:
            print("No data loaded!")
            return None
        
        # 将PyG数据转换为标签列表用于分割
        labels = [data.y.item() for data in data_list]
        
        # 数据分割
        train_indices, temp_indices = train_test_split(
            range(len(data_list)), test_size=0.4, 
            random_state=seed, stratify=labels
        )
        
        temp_labels = [labels[i] for i in temp_indices]
        val_indices, test_indices = train_test_split(
            temp_indices, test_size=0.5, 
            random_state=seed, stratify=temp_labels
        )
        
        # 创建数据子集
        train_data_list = [data_list[i] for i in train_indices]
        val_data_list = [data_list[i] for i in val_indices]
        test_data_list = [data_list[i] for i in test_indices]
        
        if save_all_results:
            visualize_graph(train_data_list, sample_index=0, title="训练样本神经元图", result_dir=result_dir)
    
    train_loader = DataLoader(train_data_list, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data_list, batch_size=32)
    test_loader = DataLoader(test_data_list, batch_size=32)
    
    # 获取特征维度
    num_features = train_data_list[0].x.shape[1]
    num_classes = len(class_names)
    
    model_class = MODEL_DICT[model_name]
    model = model_class(
        num_features=num_features,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=dropout
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=False
    )
    
    best_val_f1 = 0
    best_epoch = 0
    patience = 20  # early stopping patience
    epochs_no_improve = 0
    max_epochs = 200
    
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
    
    for epoch in range(max_epochs):
        train_metrics = train_model(model, train_loader, optimizer, device, class_weights)
        val_metrics = evaluate_model(model, val_loader, device)
        
        scheduler.step(val_metrics['f1'])
        
        for metric in train_metrics:
            history['train'][metric].append(train_metrics[metric])
        for metric in val_metrics:
            if metric not in ['predictions', 'labels']:
                history['val'][metric].append(val_metrics[metric])
        
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_epoch = epoch
            epochs_no_improve = 0
            
            if save_all_results:
                torch.save(model.state_dict(), f'{result_dir}/best_model.pth')
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            break

    if save_all_results:
        plot_training_metrics(history['train'], history['val'], result_dir=result_dir)
        plot_learning_curve(history['train'], history['val'], result_dir=result_dir)
    
    # 加载最佳模型进行测试（如果保存了模型）
    if save_all_results:
        model.load_state_dict(torch.load(f'{result_dir}/best_model.pth'))
    
    # 最终测试
    test_metrics = evaluate_model(model, test_loader, device)
    
    # 计算训练时间
    end_time = time.time()
    elapsed_time = end_time - begin_time
    
    if save_all_results:
        plot_confusion_matrix(test_metrics['labels'], test_metrics['predictions'], class_names, result_dir=result_dir)
    
    experiment_result = {
        "experiment_info": {
            "model_name": model_name,
            "seed": seed,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "total_params": total_params,
            "dataset": data_path,
            "train_size": len(train_data_list),
            "val_size": len(val_data_list),
            "test_size": len(test_data_list),
            "best_epoch": best_epoch + 1,
            "training_time": elapsed_time
        },
        "test_metrics": {
            "accuracy": float(test_metrics['accuracy']),
            "precision": float(test_metrics['precision']),
            "recall": float(test_metrics['recall']),
            "f1": float(test_metrics['f1'])
        },
        "best_val_metrics": {
            "f1": float(best_val_f1)
        },
        "history": {
            "train_loss": history['train']['loss'],
            "train_accuracy": history['train']['accuracy'],
            "train_f1": history['train']['f1'],
            "val_accuracy": history['val']['accuracy'],
            "val_f1": history['val']['f1']
        },
        "class_report": classification_report(
            test_metrics['labels'],
            test_metrics['predictions'],
            target_names=class_names,
            output_dict=True
        )
    }
    
    # 如果需要，保存详细结果
    if save_all_results:
        with open(f'{result_dir}/experiment_results.json', 'w', encoding='utf-8') as f:
            json.dump(experiment_result, f, ensure_ascii=False, indent=4)
    
    return experiment_result

def run_multiple_experiments(model_name, n_experiments=100, data_path='../../data', 
                            hidden_dim=64, dropout=0.3, save_all=False):
    
    setup_matplotlib_fonts()
    

    model_dir = f"result/{model_name}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    all_results = []
    
    print(f"开始运行 {model_name} 模型的 {n_experiments} 次实验...")
    start_time = time.time()
    
    for i in range(n_experiments):
        seed = i + 1  # 使用不同的种子
        print(f"实验 {i+1}/{n_experiments}, 种子: {seed}")
        

        result = run_single_experiment(
            model_name=model_name,
            seed=seed,
            data_path=data_path,
            hidden_dim=hidden_dim,
            dropout=dropout,
            save_all_results=save_all
        )
        
        all_results.append(result)
        

        print(f"  测试F1: {result['test_metrics']['f1']:.4f}, "
              f"准确率: {result['test_metrics']['accuracy']:.4f}, "
              f"训练时间: {result['experiment_info']['training_time']:.2f}秒")
    
    # 计算汇总统计
    accuracies = [r['test_metrics']['accuracy'] for r in all_results]
    precisions = [r['test_metrics']['precision'] for r in all_results]
    recalls = [r['test_metrics']['recall'] for r in all_results]
    f1_scores = [r['test_metrics']['f1'] for r in all_results]
    times = [r['experiment_info']['training_time'] for r in all_results]
    
    summary = {
        "model_name": model_name,
        "experiments_count": n_experiments,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "dataset": data_path,
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics_summary": {
            "accuracy": {
                "mean": np.mean(accuracies),
                "std": np.std(accuracies),
                "min": np.min(accuracies),
                "max": np.max(accuracies),
                "median": np.median(accuracies)
            },
            "precision": {
                "mean": np.mean(precisions),
                "std": np.std(precisions),
                "min": np.min(precisions),
                "max": np.max(precisions),
                "median": np.median(precisions)
            },
            "recall": {
                "mean": np.mean(recalls),
                "std": np.std(recalls),
                "min": np.min(recalls),
                "max": np.max(recalls),
                "median": np.median(recalls)
            },
            "f1": {
                "mean": np.mean(f1_scores),
                "std": np.std(f1_scores),
                "min": np.min(f1_scores),
                "max": np.max(f1_scores),
                "median": np.median(f1_scores)
            }
        },
        "training_time": {
            "mean": np.mean(times),
            "total": np.sum(times)
        },
        "all_results": all_results if save_all else "results not saved individually"
    }
    
    # 保存结果摘要
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = f"{model_dir}/summary_{timestamp}.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)
    
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\n实验完成! 总耗时: {int(hours)}小时{int(minutes)}分{int(seconds)}秒")
    print(f"结果摘要:")
    print(f"  平均准确率: {summary['metrics_summary']['accuracy']['mean']:.4f} ± {summary['metrics_summary']['accuracy']['std']:.4f}")
    print(f"  平均F1分数: {summary['metrics_summary']['f1']['mean']:.4f} ± {summary['metrics_summary']['f1']['std']:.4f}")
    print(f"  平均训练时间: {summary['training_time']['mean']:.2f}秒")
    print(f"结果保存到: {summary_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="脑神经元GNN分类实验")
    parser.add_argument('--model', type=str, default='hybrid', choices=['hybrid', 'gcn', 'gat', 'sage'],
                        help='模型类型: hybrid (混合GCN+SAGE+GAT), gcn (纯GCN), gat (纯GAT), sage(纯sage)')
    parser.add_argument('--runs', type=int, default=100, help='实验运行次数')
    parser.add_argument('--dataset', type=str, default='../../data', help='数据集路径')
    parser.add_argument('--hidden_dim', type=int, default=64, help='隐藏层维度')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout比例')
    parser.add_argument('--save_all', action='store_true', help='保存每次实验的详细结果')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"选择模型: {args.model}")
    print(f"实验次数: {args.runs}")
    print(f"数据集: {args.dataset}")
    print(f"隐藏层维度: {args.hidden_dim}")
    print(f"Dropout比例: {args.dropout}")
    print(f"保存所有详细结果: {args.save_all}")
    
    run_multiple_experiments(
        model_name=args.model,
        n_experiments=args.runs,
        data_path=args.dataset,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        save_all=args.save_all
    )

if __name__ == "__main__":
    main()