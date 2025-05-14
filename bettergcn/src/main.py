# import torch
# import numpy as np
# import random
# import os
# from process import load_data, generate_graph, split_data, create_dataset, apply_smote
# from model import ImprovedGCN
# from train import train_evaluate, plot_results
# from feature import extract_advanced_features, select_features
# from torch_geometric.loader import DataLoader
# import pathlib

# def set_seed(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#         torch.backends.cudnn.deterministic = True

# if __name__ == '__main__':
#     set_seed(42)
    
#     # 设置数据集基础路径 - 只需修改此处即可更改所有相关路径
#     dataset_name = 'EMtrace01_plus'
#     base_path = f'../datasets/{dataset_name}'
#     data_file = f'{base_path}.xlsx'
#     results_path = f'../results'
#     model_save_path = f'{results_path}/{dataset_name}_best_model.pth'
    
#     # 创建结果目录
#     os.makedirs(results_path, exist_ok=True)
    
#     # 打印当前使用的数据集和路径信息
#     print(f"使用数据集: {dataset_name}")
#     print(f"数据文件路径: {data_file}")
#     print(f"结果保存路径: {results_path}")
#     print(f"模型保存路径: {model_save_path}")
    
#     # 加载数据 
#     features, labels, encoder, class_weights = load_data(data_file)
#     train_features, test_features, train_labels, test_labels = split_data(
#         features, labels, test_size=0.2, random_state=42
#     )
    
#     # feature engineering
#     train_enhanced, pca_model = extract_advanced_features(train_features, is_train=True)
#     train_selected, selector = select_features(train_enhanced, train_labels, k=35, is_train=True)
#     test_enhanced, _ = extract_advanced_features(test_features, pca_model=pca_model, is_train=False)
#     test_selected, _ = select_features(test_enhanced, test_labels, selector=selector, is_train=False)
    
#     # apply SMOTE on training data
#     train_resampled, train_labels_resampled = apply_smote(train_selected, train_labels)
    
#     train_edge_index = generate_graph(train_resampled, k=15, threshold=None)
#     test_edge_index = generate_graph(test_selected, k=15, threshold=None)
    
#     train_dataset = create_dataset(train_resampled, train_labels_resampled, train_edge_index)
#     test_dataset = create_dataset(test_selected, test_labels, test_edge_index)
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
#     # model
#     model = ImprovedGCN(
#         num_features=35,  
#         hidden_dim=64,
#         num_classes=len(encoder.classes_), 
#         dropout=0.3
#     )
    
#     optimizer = torch.optim.AdamW(
#         model.parameters(), 
#         lr=0.001,
#         weight_decay=1e-4
#     )
    
#     criterion = torch.nn.NLLLoss()
#     train_losses, train_accs, test_accs = train_evaluate(
#         model, 
#         train_loader, 
#         test_loader, 
#         optimizer, 
#         criterion, 
#         epochs=100,
#         patience=10,
#         class_weights=class_weights
#     )
    
#     # 创建特定于当前数据集的结果目录
#     dataset_results_path = f"{results_path}/{dataset_name}"
#     os.makedirs(dataset_results_path, exist_ok=True)
    
#     # 绘制并保存训练结果
#     plot_results(train_losses, train_accs, test_accs, save_dir=dataset_results_path)
    
#     # 保存训练好的模型
#     torch.save(model.state_dict(), model_save_path)
#     print(f"模型已保存至: {model_save_path}")



import torch
import numpy as np
import random
import os
import pandas as pd
import time
from process import load_data, generate_graph, split_data, create_dataset, apply_smote
from model import ImprovedGCN
from train import train_evaluate, plot_results, plot_training_distribution, plot_prediction_results
from feature import extract_advanced_features, select_features
from torch_geometric.loader import DataLoader

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        
def run_training(seed, run_output_dir):
    """封装单次训练逻辑"""
    print(f"\n--- Starting Run with Seed: {seed} ---")
    set_seed(seed)
    os.makedirs(run_output_dir, exist_ok=True)

    # Define graph structure parameters (these should match assumptions in process.py)
    # These values will be used for reshaping flat features and for k in KNN graph generation.
    NUM_NODES_PER_GRAPH = 35 # Example: if selected features are 35, and each is a node
    NODE_FEATURE_DIM = 1     # Example: each of those 35 nodes has 1 feature
    K_FOR_GRAPH_GEN = 5      # k for KNN when generating each graph's internal edges
    THRESHOLD_FOR_GRAPH_GEN = 0.1 # Threshold for edge generation

    # load data
    # load_data now returns flat_features_per_graph, labels_encoded, encoder, class_weights_tensor, all_individual_edge_indices
    flat_features, labels, encoder, class_weights, all_edge_indices_initial = load_data(
        r'processed.csv',
        num_nodes_per_graph=NUM_NODES_PER_GRAPH,
        node_feature_dim=NODE_FEATURE_DIM,
        k_for_graph_gen=K_FOR_GRAPH_GEN,
        threshold_for_graph_gen=THRESHOLD_FOR_GRAPH_GEN
    )
    num_classes = len(encoder.classes_)

    # split_data now also splits the edge_indices_list
    train_flat_features, test_flat_features, train_labels, test_labels, train_edge_indices_initial, test_edge_indices = split_data(
        flat_features, labels, all_edge_indices_initial, test_size=0.2, random_state=seed
    )
    
    # apply SMOTE on training data (flat features and labels)
    train_flat_features_resampled, train_labels_resampled = apply_smote(
        train_flat_features, train_labels, random_state=seed
    )

    # IMPORTANT: Re-generate edge_indices for the resampled training data
    # because SMOTE changes the number and order of samples.
    train_edge_indices_resampled = []
    for i in range(train_flat_features_resampled.shape[0]):
        if NODE_FEATURE_DIM == 0:
            node_features_current_graph_np = np.empty((0,0))
        elif train_flat_features_resampled[i].size == 0:
            node_features_current_graph_np = np.empty((0,0))
        else:
            try:
                node_features_current_graph_np = train_flat_features_resampled[i].reshape(
                    NUM_NODES_PER_GRAPH, NODE_FEATURE_DIM
                )
            except ValueError as e:
                print(f"Error reshaping resampled features for graph {i} in seed {seed}: {e}. Skipping graph.")
                # Add a placeholder or skip; for simplicity, adding an empty edge_index
                # A more robust solution might involve skipping this sample or erroring out.
                train_edge_indices_resampled.append(torch.empty((2,0), dtype=torch.long))
                continue


        current_graph_edge_index = generate_graph(
            node_features_current_graph_np,
            k=K_FOR_GRAPH_GEN,
            threshold=THRESHOLD_FOR_GRAPH_GEN
        )
        train_edge_indices_resampled.append(current_graph_edge_index)
        
    try:
        train_dataset = create_dataset(
            train_flat_features_resampled, 
            train_labels_resampled, 
            train_edge_indices_resampled,
            num_nodes_per_graph=NUM_NODES_PER_GRAPH,
            node_feature_dim=NODE_FEATURE_DIM
        )
        test_dataset = create_dataset(
            test_flat_features, 
            test_labels, 
            test_edge_indices, # These are from the original split, aligned with test_flat_features
            num_nodes_per_graph=NUM_NODES_PER_GRAPH,
            node_feature_dim=NODE_FEATURE_DIM
        )
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False) # Kept batch_size consistent for now
    except ValueError as ve: # Catch specific error from create_dataset
        print(f"ValueError creating dataset/loader for seed {seed}: {ve}")
        return None
    except Exception as e:
        print(f"Error creating dataset/loader for seed {seed}: {e}")
        return None
    
    model_num_features = NODE_FEATURE_DIM
    if train_flat_features_resampled.shape[0] > 0 : # Check if any training data exists
         # If create_dataset ensures x is (num_nodes, NODE_FEATURE_DIM)
         # then model's input feature size is NODE_FEATURE_DIM
         pass # model_num_features is already NODE_FEATURE_DIM
    elif test_flat_features.shape[0] > 0: # Fallback to test data for shape if train is empty
         pass # model_num_features is already NODE_FEATURE_DIM
    else:
        print(f"No data to infer model input features for seed {seed}. Skipping run.")
        return None
    
    model = ImprovedGCN(
        num_features=model_num_features, # This should be the feature dim per node
        hidden_dim=64,
        num_classes=num_classes,
        dropout=0.3
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    # Use class_weights from load_data
    criterion = torch.nn.NLLLoss(weight=class_weights if class_weights is not None else None)


    # Train and evaluate
    try:
        train_losses, train_accs, test_accs, test_f1s, final_preds, final_labels = train_evaluate(
            model, train_loader, test_loader, optimizer, criterion,
            epochs=300, patience=30, class_weights=class_weights, # Pass class_weights here too if train_evaluate uses it
            num_classes=num_classes, encoder=encoder
        )
    except Exception as e:
         print(f"Error during training/evaluation for seed {seed}: {e}")
         return None # Skip run if training fails

    # Plot results for this specific run
    plot_results(train_losses, train_accs, test_accs, test_f1s, save_dir=run_output_dir)
    plot_prediction_results(final_labels, final_preds, encoder, save_dir=run_output_dir)
    torch.save(model.state_dict(), os.path.join(run_output_dir, 'best_model.pth'))
    print(f"Run {seed}: Best model saved to {os.path.join(run_output_dir, 'best_model.pth')}")

    # Return key results for aggregation
    best_test_f1 = max(test_f1s) if test_f1s else 0
    best_test_acc = max(test_accs) if test_accs else 0
    final_epoch = len(train_losses)
    return {
        'seed': seed,
        'best_test_f1': best_test_f1,
        'best_test_acc': best_test_acc,
        'final_epoch': final_epoch,
        'output_dir': run_output_dir
    }
    
if __name__ == '__main__':
    num_runs = 1 # Or 1 if you want to test a single run first
    base_seed = 42
    all_runs_results = []
    multi_run_base_dir = f'results/multi_run_{time.strftime("%Y%m%d-%H%M%S")}'
    os.makedirs(multi_run_base_dir, exist_ok=True)

    for i in range(num_runs):
        current_seed = base_seed + i
        run_output_dir = os.path.join(multi_run_base_dir, f'run_{i+1}_seed_{current_seed}')
        run_result = run_training(current_seed, run_output_dir)
        if run_result:
            all_runs_results.append(run_result)

    # --- Multi-run Analysis ---
    if all_runs_results:
        print("\n--- Multi-Run Analysis ---")
        results_df = pd.DataFrame(all_runs_results)
        print(results_df)
        results_df.to_csv(os.path.join(multi_run_base_dir, 'multi_run_summary.csv'), index=False)

        analysis_plot_dir = os.path.join(multi_run_base_dir, 'analysis_plots')
        os.makedirs(analysis_plot_dir, exist_ok=True) # Ensure this directory exists
        plot_training_distribution(all_runs_results, metric='best_test_f1', save_dir=analysis_plot_dir)
        plot_training_distribution(all_runs_results, metric='best_test_acc', save_dir=analysis_plot_dir)
    else:
        print("No successful runs completed for multi-run analysis.")

    print(f"\nMulti-run execution finished. Results stored in: {multi_run_base_dir}")