"""
Topology Clustering Analysis Module

This module performs clustering analysis on topology matrices to identify
temporal patterns in neuron connectivity. It supports multiple clustering algorithms
and provides visualization tools for analysis.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
from typing import Tuple, List, Optional, Dict, Any
from abc import ABC, abstractmethod
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ClusteringAlgorithm(ABC):
    """Abstract base class for clustering algorithms."""
    
    @abstractmethod
    def determine_parameters(self, X: np.ndarray) -> Dict[str, Any]:
        """Determine optimal parameters for the clustering algorithm."""
        pass
    
    @abstractmethod
    def fit_predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Perform clustering and return labels."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the name of the algorithm."""
        pass

class KMeansClusterer(ClusteringAlgorithm):
    """KMeans clustering implementation."""
    
    def __init__(self, max_k: int = 10, default_n_clusters: int = 6):
        """
        Initialize KMeans clusterer.

        Args:
            max_k: Maximum number of clusters to test
            default_n_clusters: Default number of clusters if optimal k is not determined
        """
        self.max_k = max_k
        self.default_n_clusters = default_n_clusters
    
    def determine_parameters(self, X: np.ndarray) -> Dict[str, Any]:
        inertia = []
        silhouette = []
        K_range = range(2, self.max_k + 1)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            inertia.append(kmeans.inertia_)
            score = silhouette_score(X, kmeans.labels_)
            silhouette.append(score)
        
        # Plot parameter selection graphs
        self._plot_parameter_selection(K_range, inertia, silhouette)
        
        # Return optimal k based on silhouette score
        optimal_k = K_range[np.argmax(silhouette)]
        return {"n_clusters": optimal_k}
    
    def fit_predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        n_clusters = kwargs.get("n_clusters", self.default_n_clusters)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        return kmeans.fit_predict(X)
    
    def get_name(self) -> str:
        return "KMeans"
    
    def _plot_parameter_selection(self, K_range, inertia, silhouette):
        fig, ax1 = plt.subplots(figsize=(10, 5))
        color = 'tab:blue'
        ax1.set_xlabel('Number of Clusters (K)')
        ax1.set_ylabel('Inertia', color=color)
        ax1.plot(K_range, inertia, 'o-', color=color, label='Inertia')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend(loc='upper left')
        
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Silhouette Score', color=color)
        ax2.plot(K_range, silhouette, 's--', color=color, label='Silhouette Score')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc='upper right')
        
        plt.title('KMeans Parameter Selection')
        plt.show()

class DBSCANClusterer(ClusteringAlgorithm):
    """DBSCAN clustering implementation."""
    
    def __init__(self, k: int = 4, default_eps: float = 0.5):
        """
        Initialize DBSCAN clusterer.

        Args:
            k: Number of neighbors for k-distance graph (min_samples = k + 1)
            default_eps: Default eps value if not determined from k-distance graph
        """
        self.k = k
        self.default_eps = default_eps
    
    def determine_parameters(self, X: np.ndarray) -> Dict[str, Any]:
        # Determine eps using k-distance graph
        neighbors = NearestNeighbors(n_neighbors=self.k)
        neighbors_fit = neighbors.fit(X)
        distances, _ = neighbors_fit.kneighbors(X)
        distances = np.sort(distances[:, self.k-1])
        
        # Plot k-distance graph
        plt.figure(figsize=(10, 5))
        plt.plot(distances)
        plt.ylabel(f'{self.k}-distance')
        plt.xlabel('Points sorted by distance')
        plt.title(f'k-distance Graph (k={self.k}) for DBSCAN eps Selection')
        plt.show()
        
        # Estimate eps from the "elbow" in the k-distance graph
        eps = np.percentile(distances, 90)
        
        return {
            "eps": eps,
            "min_samples": self.k + 1
        }
    
    def fit_predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        eps = kwargs.get("eps", self.default_eps)
        min_samples = kwargs.get("min_samples", self.k + 1)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        return dbscan.fit_predict(X)
    
    def get_name(self) -> str:
        return "DBSCAN"

class ClusteringFactory:
    """Factory class for creating clustering algorithm instances."""
    
    @staticmethod
    def get_algorithm(algorithm_id: int, **kwargs) -> ClusteringAlgorithm:
        """
        Get clustering algorithm instance.

        Args:
            algorithm_id: ID of the algorithm (1: KMeans, 2: DBSCAN)
            **kwargs: Algorithm-specific parameters
                For KMeans: max_k, default_n_clusters
                For DBSCAN: k, default_eps
        """
        algorithms = {
            1: KMeansClusterer(
                max_k=kwargs.get('max_k', 10),
                default_n_clusters=kwargs.get('default_n_clusters', 6)
            ),
            2: DBSCANClusterer(
                k=kwargs.get('k', 4),
                default_eps=kwargs.get('default_eps', 0.5)
            )
        }
        
        if algorithm_id not in algorithms:
            raise ValueError(f"Algorithm ID {algorithm_id} not supported")
        
        return algorithms[algorithm_id]

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

def visualize_clusters(
    X: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
    feature_cols: List[str],
    algorithm_name: str,
    interactive: bool = True
) -> None:
    """
    Visualize clustering results using PCA for dimensionality reduction.

    Args:
        X: Preprocessed feature matrix
        labels: Cluster labels
        timestamps: Array of timestamps
        feature_cols: List of feature column names
        algorithm_name: Name of the clustering algorithm
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
    
    title = f'{algorithm_name} Clustering Results (PCA)'
    
    if interactive:
        # Create interactive plot with Plotly
        fig = px.scatter(
            viz_df,
            x='PCA1',
            y='PCA2',
            color='Cluster',
            title=title,
            hover_data=['Time_Stamp']
        )
        fig.show()
    else:
        # Create static plot with matplotlib
        plt.figure(figsize=(12, 8))
        scatter = sns.scatterplot(
            data=viz_df,
            x='PCA1',
            y='PCA2',
            hue='Cluster',
            palette='Set1' if algorithm_name != 'DBSCAN' else 'Set2',
            s=100,
            alpha=0.7
        )
        
        # Special handling for DBSCAN noise points
        if algorithm_name == 'DBSCAN':
            noise_mask = labels == -1
            if noise_mask.any():
                plt.scatter(
                    viz_df.loc[noise_mask, 'PCA1'],
                    viz_df.loc[noise_mask, 'PCA2'],
                    color='grey',
                    marker='x',
                    label='Noise',
                    alpha=0.5
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
        plt.title(title)
        plt.legend(title='Cluster')
        plt.tight_layout()
        plt.show()

def analyze_clusters(labels: np.ndarray, timestamps: np.ndarray, algorithm_name: str) -> None:
    """
    Analyze and print the timestamps belonging to each cluster.

    Args:
        labels: Cluster labels
        timestamps: Array of timestamps
        algorithm_name: Name of the clustering algorithm
    """
    unique_labels = np.unique(labels)
    for label in unique_labels:
        if algorithm_name == 'DBSCAN' and label == -1:
            cluster_name = "Noise points"
        else:
            cluster_name = f"{algorithm_name} Cluster {label}"
            
        cluster_timestamps = timestamps[labels == label]
        logging.info(f"\n{cluster_name} timestamps:")
        logging.info(cluster_timestamps)

def main(
    file_path: str = '../datasets/Day9_topology_matrix.xlsx',
    algorithm_id: int = 1,
    **algorithm_params
) -> None:
    """
    Execute the complete clustering analysis workflow.

    Args:
        file_path: Path to the topology matrix Excel file
        algorithm_id: ID of the clustering algorithm to use (1: KMeans, 2: DBSCAN)
        **algorithm_params: Algorithm-specific parameters
            For KMeans: max_k, default_n_clusters
            For DBSCAN: k, default_eps
    """
    try:
        # Get the selected clustering algorithm with custom parameters
        algorithm = ClusteringFactory.get_algorithm(algorithm_id, **algorithm_params)
        logging.info(f"Using {algorithm.get_name()} clustering algorithm")
        
        # Load and preprocess data
        X_scaled, timestamps, feature_cols = load_and_preprocess_data(file_path)
        
        # Determine algorithm parameters
        params = algorithm.determine_parameters(X_scaled)
        logging.info(f"Determined parameters: {params}")
        
        # Perform clustering
        labels = algorithm.fit_predict(X_scaled, **params)
        
        # Visualize results
        visualize_clusters(X_scaled, labels, timestamps, feature_cols, 
                         algorithm.get_name(), interactive=True)
        visualize_clusters(X_scaled, labels, timestamps, feature_cols,
                         algorithm.get_name(), interactive=False)
        
        # Analyze clusters
        analyze_clusters(labels, timestamps, algorithm.get_name())
        
    except Exception as e:
        logging.error(f"Error in clustering analysis: {e}")
        raise

if __name__ == "__main__":
    main(algorithm_id=1)  # Default to KMeans