import torch
from process import load_data, generate_graph, split_data, create_dataset
from model import GCN
from train import train_evaluate, plot_results, setup_matplotlib_fonts, plot_confusion_matrix, plot_training_metrics, plot_learning_curve
from torch_geometric.data import DataLoader

if __name__ == '__main__':
    # 设置matplotlib字体和样式
    setup_matplotlib_fonts()
    
    features, labels, encoder = load_data('dataset\processed_EMtrace01_plus.xlsx')
    # features_resampled, labels_resampled = apply_smote(features, labels)
    train_features, test_features, train_labels, test_labels = split_data(
        features, labels, test_size=0.2, random_state=42
    )
    edge_index = generate_graph(train_features, threshold=0.4)

    train = create_dataset(train_features, train_labels, edge_index)
    test = create_dataset(test_features, test_labels, edge_index)
    train_loader = DataLoader(train, batch_size=32, shuffle=True)
    test_loader = DataLoader(test, batch_size=32, shuffle=False)

    model = GCN(num_features=1, hidden_dim=16, num_classes=len(encoder.classes_), use_batch_norm=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.008, weight_decay=1e-4)
    criterion = torch.nn.NLLLoss()
    
    # 使用新的训练函数，获取详细指标
    train_metrics, val_metrics, final_predictions, final_labels = train_evaluate(
        model, train_loader, test_loader, optimizer, criterion, epochs=50
    )

    # 绘制各种图表
    result_dir = 'results'
    
    # 1. 原始的简单结果图
    plot_results(train_metrics['loss'], train_metrics['accuracy'], val_metrics['accuracy'], result_dir)
    
    # 2. 详细的训练指标图
    plot_training_metrics(train_metrics, val_metrics, result_dir)
    
    # 3. 学习曲线
    plot_learning_curve(train_metrics, val_metrics, result_dir)
    
    # 4. 混淆矩阵
    class_names = encoder.classes_
    plot_confusion_matrix(final_labels, final_predictions, class_names, result_dir)
    
    print(f"\n训练完成！所有图表已保存到 {result_dir} 目录")
    print(f"最终测试准确率: {val_metrics['accuracy'][-1]:.4f}")
    print(f"最终测试F1分数: {val_metrics['f1'][-1]:.4f}")