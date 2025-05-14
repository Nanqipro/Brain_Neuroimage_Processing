import torch
import numpy as np
import random
import os
from process import load_data, generate_graph, split_data, create_dataset, apply_smote
from model import ImprovedGCN
from train import train_evaluate, plot_results
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

if __name__ == '__main__':
    set_seed(42)
    os.makedirs('../results', exist_ok=True)
    
    # load data 
    features, labels, encoder, class_weights = load_data('../dataset/EMtrace01.xlsx')
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
        epochs=150,
        patience=20,
        class_weights=class_weights
    )
    
    plot_results(train_losses, train_accs, test_accs)
    
    torch.save(model.state_dict(), '../results/EMtrace01_best_model.pth')