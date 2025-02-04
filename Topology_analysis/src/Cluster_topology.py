"""
Topology Clustering Analysis Module

This module performs clustering analysis on topology matrices to identify
temporal patterns in neuron connectivity. It includes:
1. Data preprocessing and standardization
2. Optimal cluster number determination
3. K-means clustering
4. Dimensionality reduction for visualization
5. Interactive visualization of clustering results
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import plotly.express as px
from typing import Tuple, List, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_and_preprocess_data(file_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load and preprocess topology matrix data.

    Args:
        file_path: Path to the Excel file containing topology matrix

    Returns:
        Tuple containing:
        - Preprocessed feature matrix
        - Array of timestamps
        - List of feature column names

    Raises:
        FileNotFoundError: If the specified file doesn't exist
        ValueError: If data is invalid or contains unexpected values
    """
    try:
        df = pd.read_excel(file_path)
        logging.info(f"Loaded data from {file_path}")
        
        # Separate timestamps and features
        timestamp_col = 'Time_Stamp'
        feature_cols = [col for col in df.columns if col != timestamp_col]
        
        X = df[feature_cols].values
        timestamps = df[timestamp_col].values
        
        # Handle missing values
        if np.isnan(X).any():
            logging.warning("Found missing values, filling with zeros")
            X = np.nan_to_num(X)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, timestamps, feature_cols
    except Exception as e:
        logging.error(f"Error in data preprocessing: {e}")
        raise

def determine_optimal_k(X: np.ndarray, max_k: int = 10) -> Tuple[List[float], List[float]]:
    """
    Determine optimal number of clusters using elbow method and silhouette score.

    Args:
        X: Preprocessed feature matrix
        max_k: Maximum number of clusters to test

    Returns:
        Tuple containing:
        - List of inertia values
        - List of silhouette scores
    """
    inertia = []
    silhouette = []
    K_range = range(2, max_k + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
        score = silhouette_score(X, kmeans.labels_)
        silhouette.append(score)
    
    # Plot elbow curve and silhouette scores
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # Plot inertia
    color = 'tab:blue'
    ax1.set_xlabel('Number of Clusters (K)')
    ax1.set_ylabel('Inertia', color=color)
    ax1.plot(K_range, inertia, 'o-', color=color, label='Inertia')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')
    
    # Plot silhouette score
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Silhouette Score', color=color)
    ax2.plot(K_range, silhouette, 's--', color=color, label='Silhouette Score')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')
    
    plt.title('Optimal Cluster Number Determination')
    plt.show()
    
    return inertia, silhouette

def perform_clustering(X: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Perform K-means clustering on the preprocessed data.

    Args:
        X: Preprocessed feature matrix
        n_clusters: Number of clusters to form

    Returns:
        Array of cluster labels
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(X)

def visualize_clusters(
    X: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
    feature_cols: List[str],
    interactive: bool = True
) -> None:
    """
    Visualize clustering results using PCA for dimensionality reduction.

    Args:
        X: Preprocessed feature matrix
        labels: Cluster labels
        timestamps: Array of timestamps
        feature_cols: List of feature column names
        interactive: Whether to create interactive plot
    """
    # Perform PCA
    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X)
    
    # Create DataFrame for visualization
    viz_df = pd.DataFrame({
        'PCA1': X_pca[:, 0],
        'PCA2': X_pca[:, 1],
        'Cluster': labels,
        'Time_Stamp': timestamps
    })
    
    if interactive:
        # Create interactive plot with Plotly
        fig = px.scatter(
            viz_df,
            x='PCA1',
            y='PCA2',
            color='Cluster',
            title='Cluster Visualization (PCA)',
            hover_data=['Time_Stamp']
        )
        fig.show()
    else:
        # Create static plot with matplotlib
        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            data=viz_df,
            x='PCA1',
            y='PCA2',
            hue='Cluster',
            palette='Set1',
            s=100,
            alpha=0.7
        )
        
        # Add labels for selected points
        label_step = max(1, len(viz_df) // 20)
        for i in range(0, len(viz_df), label_step):
            plt.text(
                viz_df['PCA1'].iloc[i] + 0.02,
                viz_df['PCA2'].iloc[i] + 0.02,
                str(viz_df['Time_Stamp'].iloc[i]),
                fontsize=9
            )
        
        plt.xlabel('PCA1')
        plt.ylabel('PCA2')
        plt.title('Cluster Visualization (PCA)')
        plt.legend(title='Cluster')
        plt.tight_layout()
        plt.show()

def analyze_clusters(labels: np.ndarray, timestamps: np.ndarray) -> None:
    """
    Analyze and print the timestamps belonging to each cluster.

    Args:
        labels: Cluster labels
        timestamps: Array of timestamps
    """
    for cluster in range(labels.max() + 1):
        cluster_timestamps = timestamps[labels == cluster]
        logging.info(f"\nCluster {cluster} timestamps:")
        logging.info(cluster_timestamps)

def main(file_path: str = '../datasets/Day9_topology_matrix.xlsx', n_clusters: int = 6) -> None:
    """
    Execute the complete clustering analysis workflow.

    Args:
        file_path: Path to the topology matrix Excel file
        n_clusters: Number of clusters to form
    """
    try:
        # Load and preprocess data
        X_scaled, timestamps, feature_cols = load_and_preprocess_data(file_path)
        
        # Determine optimal number of clusters
        determine_optimal_k(X_scaled, max_k=10)
        
        # Perform clustering
        labels = perform_clustering(X_scaled, n_clusters)
        
        # Visualize results
        visualize_clusters(X_scaled, labels, timestamps, feature_cols, interactive=True)
        visualize_clusters(X_scaled, labels, timestamps, feature_cols, interactive=False)
        
        # Analyze clusters
        analyze_clusters(labels, timestamps)
        
    except Exception as e:
        logging.error(f"Error in clustering analysis: {e}")
        raise

if __name__ == "__main__":
    main()