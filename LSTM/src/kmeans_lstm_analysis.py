import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
import warnings
import os
warnings.filterwarnings('ignore')

# Path Configuration
class Config:
    def __init__(self):
        # Base directory
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Data paths
        self.data_dir = os.path.join(self.base_dir, 'datasets')
        self.data_file = os.path.join(self.data_dir, 'Day6_with_behavior_labels_filled.xlsx')
        
        # Output paths
        self.output_dir = os.path.join(self.base_dir, 'results')
        self.model_dir = os.path.join(self.base_dir, 'models')
        
        # Result files
        self.loss_plot = os.path.join(self.output_dir, 'training_loss.png')
        self.cluster_plot = os.path.join(self.output_dir, 'cluster_visualization.png')
        self.model_path = os.path.join(self.model_dir, 'neuron_lstm_model.pth')
        
        # Model parameters
        self.sequence_length = 10
        self.hidden_size = 128
        self.num_layers = 2
        self.batch_size = 32
        self.learning_rate = 0.001
        self.num_epochs = 50
        self.n_clusters = 5
        self.test_size = 0.2
        self.random_seed = 42
        
    def setup_directories(self):
        """Create necessary directories"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
    def validate_paths(self):
        """Validate required files and directories"""
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"Data file not found: {self.data_file}")

# Set random seed
def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

# Data loading and preprocessing class
class NeuronDataProcessor:
    def __init__(self, config):
        self.config = config
        self.data = pd.read_excel(config.data_file)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def preprocess_data(self):
        # Extract neuron columns
        neuron_cols = [f'n{i}' for i in range(1, 63)]
        
        # Check available columns
        available_cols = [col for col in neuron_cols if col in self.data.columns]
        print(f"Total neurons: {len(neuron_cols)}")
        print(f"Available neurons: {len(available_cols)}")
        print(f"Missing neurons: {set(neuron_cols) - set(available_cols)}")
        
        # Get only available neuron data
        X = self.data[available_cols].values
        
        # Handle missing values
        if np.isnan(X).any():
            print("Found missing values, filling with mean values")
            # Fill missing values with mean of each column
            X = np.nan_to_num(X, nan=np.nanmean(X))
        
        # Standardize neuron data
        X_scaled = self.scaler.fit_transform(X)
        
        # Encode behavior labels
        if 'behavior' not in self.data.columns:
            raise ValueError("Behavior column not found in the dataset")
            
        # Handle missing behavior labels
        behavior_data = self.data['behavior'].fillna('unknown')
        y = self.label_encoder.fit_transform(behavior_data)
        
        # Print label encoding information
        label_mapping = dict(zip(self.label_encoder.classes_, 
                               self.label_encoder.transform(self.label_encoder.classes_)))
        print("\nBehavior label mapping:")
        for label, code in label_mapping.items():
            count = sum(y == code)
            print(f"{label}: {code} (Count: {count})")
        
        return X_scaled, y

    def apply_kmeans(self, X):
        # Apply K-means clustering
        print(f"\nApplying K-means clustering with {self.config.n_clusters} clusters")
        kmeans = KMeans(n_clusters=self.config.n_clusters, 
                       random_state=self.config.random_seed,
                       n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        # Print cluster distribution
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        print("\nCluster distribution:")
        for label, count in zip(unique_labels, counts):
            print(f"Cluster {label}: {count} samples")
        
        return kmeans, cluster_labels

# LSTM dataset class
class NeuronDataset(Dataset):
    def __init__(self, X, y, sequence_length):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.X) - self.sequence_length
        
    def __getitem__(self, idx):
        return (self.X[idx:idx+self.sequence_length], 
                self.y[idx+self.sequence_length-1])

# LSTM model class
class NeuronLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(NeuronLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Training function
def train_model(model, train_loader, criterion, optimizer, device, num_epochs, config):
    model.train()
    train_losses = []
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
            
    return train_losses

def plot_training_loss(train_losses, config):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(config.loss_plot)
    plt.close()

def plot_clusters(X_scaled, cluster_labels, config):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis')
    plt.title('Neuron Activity Clusters')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.colorbar(scatter)
    plt.savefig(config.cluster_plot)
    plt.close()

def main():
    # Initialize configuration
    config = Config()
    
    try:
        # Validate and create necessary directories
        config.validate_paths()
        config.setup_directories()
        
        # Set random seed
        set_random_seed(config.random_seed)
        
        # Data preprocessing
        processor = NeuronDataProcessor(config)
        X_scaled, y = processor.preprocess_data()
        
        # Apply K-means clustering
        kmeans, cluster_labels = processor.apply_kmeans(X_scaled)
        
        # Add clustering results to features
        X_with_clusters = np.column_stack((X_scaled, cluster_labels))
        
        # Split training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_with_clusters, y, test_size=config.test_size, random_state=config.random_seed
        )
        
        # Create data loader
        train_dataset = NeuronDataset(X_train, y_train, config.sequence_length)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size, 
            shuffle=True
        )
        
        # Initialize model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_size = X_with_clusters.shape[1]
        num_classes = len(np.unique(y))
        model = NeuronLSTM(
            input_size, 
            config.hidden_size, 
            config.num_layers, 
            num_classes
        ).to(device)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        
        # Train model
        train_losses = train_model(
            model, 
            train_loader, 
            criterion, 
            optimizer, 
            device, 
            config.num_epochs,
            config
        )
        
        # Plot training loss
        plot_training_loss(train_losses, config)
        
        # Save model
        torch.save(model.state_dict(), config.model_path)
        
        # Visualize clustering results
        plot_clusters(X_scaled, cluster_labels, config)
        
        print("Analysis completed! Results saved to:")
        print(f"Training loss plot: {config.loss_plot}")
        print(f"Cluster visualization: {config.cluster_plot}")
        print(f"Model file: {config.model_path}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main() 