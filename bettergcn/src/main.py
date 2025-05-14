import torch
import numpy as np
import random
import os
from process import load_data, generate_graph, split_data, create_dataset, apply_smote
from model import ImprovedGCN
from train import train_evaluate, plot_results
from feature import extract_advanced_features, select_features
from torch_geometric.loader import DataLoader
import pathlib

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    set_seed(42)
    
    # 设置数据集基础路径 - 只需修改此处即可更改所有相关路径
    dataset_name = 'EMtrace01_plus'
    base_path = f'../datasets/{dataset_name}'
    data_file = f'{base_path}.xlsx'
    results_path = f'../results'
    model_save_path = f'{results_path}/{dataset_name}_best_model.pth'
    
    # 创建结果目录
    os.makedirs(results_path, exist_ok=True)
    
    # 打印当前使用的数据集和路径信息
    print(f"使用数据集: {dataset_name}")
    print(f"数据文件路径: {data_file}")
    print(f"结果保存路径: {results_path}")
    print(f"模型保存路径: {model_save_path}")
    
    # 加载数据 
    features, labels, encoder, class_weights = load_data(data_file)
    train_features, test_features, train_labels, test_labels = split_data(
        features, labels, test_size=0.2, random_state=42
    )
    
    # feature engineering
    train_enhanced, pca_model = extract_advanced_features(train_features, is_train=True)
    train_selected, selector = select_features(train_enhanced, train_labels, k=40, is_train=True)
    test_enhanced, _ = extract_advanced_features(test_features, pca_model=pca_model, is_train=False)
    test_selected, _ = select_features(test_enhanced, test_labels, selector=selector, is_train=False)
    
    # apply SMOTE on training data
    train_resampled, train_labels_resampled = apply_smote(train_selected, train_labels)
    
    train_edge_index = generate_graph(train_resampled, k=15, threshold=None)
    test_edge_index = generate_graph(test_selected, k=15, threshold=None)
    
    train_dataset = create_dataset(train_resampled, train_labels_resampled, train_edge_index)
    test_dataset = create_dataset(test_selected, test_labels, test_edge_index)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # model
    model = ImprovedGCN(
        num_features=40,  
        hidden_dim=64,
        num_classes=len(encoder.classes_), 
        dropout=0.3
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=0.001,
        weight_decay=1e-4
    )
    
    criterion = torch.nn.NLLLoss()
    train_losses, train_accs, test_accs = train_evaluate(
        model, 
        train_loader, 
        test_loader, 
        optimizer, 
        criterion, 
        epochs=100,
        patience=10,
        class_weights=class_weights
    )
    
    # 创建特定于当前数据集的结果目录
    dataset_results_path = f"{results_path}/{dataset_name}"
    os.makedirs(dataset_results_path, exist_ok=True)
    
    # 绘制并保存训练结果
    plot_results(train_losses, train_accs, test_accs, save_dir=dataset_results_path)
    
    # 保存训练好的模型
    torch.save(model.state_dict(), model_save_path)
    print(f"模型已保存至: {model_save_path}")