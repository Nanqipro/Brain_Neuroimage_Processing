import torch
from process import load_data, generate_graph, split_data, create_dataset
from model import GCN
from train import train_evaluate, plot_results
from torch_geometric.data import DataLoader

if __name__ == '__main__':

    features, labels, encoder = load_data(r'dataset\Emtrace\processed.csv')
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
    train_losses, train_accs, test_accs = train_evaluate(model, train_loader, test_loader, optimizer, criterion, epochs=50)

    plot_results(train_losses, train_accs, test_accs)