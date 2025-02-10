"""
Neuron State Clustering Analysis Module

This module performs clustering analysis on neuron activation states to:
1. Identify similar neuron activation patterns
2. Group timestamps into different behavioral patterns
3. Analyze behavior label distribution for each cluster
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
from typing import Tuple, List, Dict, Any
from abc import ABC, abstractmethod
import logging
import os

# ===================== Configuration =====================
# Base path configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'datasets')
OUTPUT_DIR = os.path.join(BASE_DIR, 'datasets')

# File path configuration
DEFAULT_FILES = {
    'day3': {
        'states': os.path.join(DATA_DIR, 'Day3_neuron_states.xlsx'),
        'behavior': os.path.join(DATA_DIR, 'Day3_with_behavior_labels_filled.xlsx')
    },
    'day6': {
        'states': os.path.join(DATA_DIR, 'Day6_neuron_states.xlsx'),
        'behavior': os.path.join(DATA_DIR, 'Day6_with_behavior_labels_filled.xlsx')
    },
    'day9': {
        'states': os.path.join(DATA_DIR, 'Day9_neuron_states.xlsx'),
        'behavior': os.path.join(DATA_DIR, 'Day9_with_behavior_labels_filled.xlsx')
    }
}
# ===================== End Configuration =====================

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
            max_k: Maximum number of clusters to test in parameter selection
            default_n_clusters: Default number of clusters to use
        """
        self.max_k = max_k
        self.default_n_clusters = default_n_clusters
    
    def determine_parameters(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Determine optimal parameters using elbow method and silhouette score.
        """
        inertia = []
        silhouette = []
        K_range = range(2, self.max_k + 1)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            inertia.append(kmeans.inertia_)
            score = silhouette_score(X, kmeans.labels_)
            silhouette.append(score)
        
        # Plot parameter selection visualization
        self._plot_parameter_selection(K_range, inertia, silhouette)
        
        # Return the default number of clusters
        return {"n_clusters": self.default_n_clusters}
    
    def fit_predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Perform KMeans clustering."""
        n_clusters = kwargs.get("n_clusters", self.default_n_clusters)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        logging.info(f"KMeans clustering completed, silhouette score: {score:.3f}")
        return labels
    
    def get_name(self) -> str:
        return "KMeans"
    
    def _plot_parameter_selection(self, K_range, inertia, silhouette):
        """Plot elbow curve and silhouette scores."""
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
        
        plt.title('KMeans Parameter Selection')
        plt.show()

class DBSCANClusterer(ClusteringAlgorithm):
    """DBSCAN clustering implementation."""
    
    def __init__(self, k: int = 4, default_eps: float = 0.5):
        """
        Initialize DBSCAN clusterer.

        Args:
            k: Number of neighbors for k-distance graph
            default_eps: Default epsilon value
        """
        self.k = k
        self.default_eps = default_eps
    
    def determine_parameters(self, X: np.ndarray) -> Dict[str, Any]:
        """Determine optimal parameters using k-distance graph."""
        neighbors = NearestNeighbors(n_neighbors=self.k)
        neighbors_fit = neighbors.fit(X)
        distances, _ = neighbors_fit.kneighbors(X)
        distances = np.sort(distances[:, self.k-1])
        
        # Plot k-distance graph
        plt.figure(figsize=(10, 5))
        plt.plot(distances)
        plt.ylabel(f'{self.k}-distance')
        plt.xlabel('Points Sorted by Distance')
        plt.title(f'k-distance Graph (k={self.k}) for DBSCAN')
        plt.show()
        
        # Estimate eps from the k-distance graph
        eps = np.percentile(distances, 90)
        
        return {
            "eps": eps,
            "min_samples": self.k + 1
        }
    
    def fit_predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Perform DBSCAN clustering."""
        eps = kwargs.get("eps", self.default_eps)
        min_samples = kwargs.get("min_samples", self.k + 1)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        
        # Calculate silhouette score (excluding noise points)
        mask = labels != -1
        if mask.any():
            score = silhouette_score(X[mask], labels[mask])
            logging.info(f"DBSCAN clustering completed, silhouette score: {score:.3f}")
        else:
            logging.warning("All points classified as noise")
        
        return labels
    
    def get_name(self) -> str:
        return "DBSCAN"

class ClusteringFactory:
    """Factory class for creating clustering algorithm instances."""
    
    @staticmethod
    def get_algorithm(algorithm_id: int, **kwargs) -> ClusteringAlgorithm:
        """
        Get clustering algorithm instance.

        Args:
            algorithm_id: Algorithm identifier (1: KMeans, 2: DBSCAN)
            **kwargs: Algorithm-specific parameters
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

def load_data(states_file: str, behavior_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load neuron state data and behavior label data.

    Args:
        states_file: Path to neuron state data file
        behavior_file: Path to behavior label data file

    Returns:
        Tuple of neuron state DataFrame and behavior label DataFrame
    """
    try:
        # Load neuron state data
        states_df = pd.read_excel(states_file)
        logging.info(f"Successfully loaded neuron state data: {states_file}")

        # Load behavior label data
        behavior_df = pd.read_excel(behavior_file)
        logging.info(f"Successfully loaded behavior label data: {behavior_file}")

        return states_df, behavior_df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def preprocess_data(states_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Preprocess neuron state data.

    Args:
        states_df: Neuron state DataFrame

    Returns:
        Tuple of feature matrix, timestamp array, and feature column names
    """
    try:
        # Separate timestamps and neuron states
        timestamp_col = 'Time_Stamp'
        neuron_cols = [col for col in states_df.columns if col != timestamp_col]
        
        # Extract feature matrix and timestamps
        X = states_df[neuron_cols].values
        timestamps = states_df[timestamp_col].values
        
        logging.info(f"Data preprocessing completed, feature count: {len(neuron_cols)}")
        return X, timestamps, neuron_cols
    except Exception as e:
        logging.error(f"Error in data preprocessing: {e}")
        raise

def visualize_clusters(X: np.ndarray, labels: np.ndarray, timestamps: np.ndarray,
                      algorithm_name: str, title: str = None) -> None:
    """
    Visualize clustering results using PCA.

    Args:
        X: Feature matrix
        labels: Cluster labels
        timestamps: Timestamp array
        algorithm_name: Clustering algorithm name
        title: Plot title
    """
    # Use PCA for dimensionality reduction to 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Create visualization DataFrame
    viz_df = pd.DataFrame({
        'PCA1': X_pca[:, 0],
        'PCA2': X_pca[:, 1],
        'Cluster': labels,
        'Timestamp': timestamps
    })
    
    # Create scatter plot
    plt.figure(figsize=(12, 8))
    
    # Handle noise points for DBSCAN
    if algorithm_name == "DBSCAN":
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
            viz_df = viz_df[~noise_mask]  # Remove noise points for main plot
    
    scatter = sns.scatterplot(
        data=viz_df,
        x='PCA1',
        y='PCA2',
        hue='Cluster',
        palette='deep',
        s=100,
        alpha=0.7
    )
    
    # Add timestamp labels
    label_step = max(1, len(viz_df) // 20)
    for i in range(0, len(viz_df), label_step):
        plt.text(
            viz_df['PCA1'].iloc[i] + 0.02,
            viz_df['PCA2'].iloc[i] + 0.02,
            str(viz_df['Timestamp'].iloc[i]),
            fontsize=9
        )
    
    plt.title(title or f"{algorithm_name} Clustering Results")
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.show()

def analyze_cluster_behaviors(labels: np.ndarray, timestamps: np.ndarray, 
                            behavior_df: pd.DataFrame, algorithm_name: str) -> None:
    """
    Analyze behavior label distribution for each cluster.

    Args:
        labels: Cluster labels
        timestamps: Timestamp array
        behavior_df: Behavior label DataFrame
        algorithm_name: Clustering algorithm name
    """
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        # Get cluster name
        if algorithm_name == "DBSCAN" and label == -1:
            cluster_name = "Noise Points"
        else:
            cluster_name = f"Cluster {label}"
        
        # Get timestamps for this cluster
        cluster_timestamps = timestamps[labels == label]
        
        # Get behavior labels for this cluster
        cluster_behaviors = behavior_df[behavior_df['stamp'].isin(cluster_timestamps)]['behavior']
        behavior_counts = cluster_behaviors.value_counts()
        behavior_percentages = (behavior_counts / len(cluster_timestamps) * 100).round(2)
        
        # Output analysis results
        logging.info(f"\n{cluster_name} Analysis Results:")
        logging.info(f"Contains {len(cluster_timestamps)} timestamps")
        logging.info("\nBehavior Distribution:")
        for behavior, percentage in behavior_percentages.items():
            logging.info(f"- {behavior}: {percentage}% ({behavior_counts[behavior]} occurrences)")
        
        # Output timestamps
        logging.info("\nTimestamps:")
        timestamps_str = ', '.join(map(str, sorted(cluster_timestamps)))
        logging.info(timestamps_str)

def main(day: str = 'day6', algorithm_id: int = 1, **algorithm_params) -> None:
    """
    Execute complete clustering analysis workflow.

    Args:
        day: Data date ('day3', 'day6', 'day9')
        algorithm_id: Clustering algorithm ID (1: KMeans, 2: DBSCAN)
        **algorithm_params: Algorithm-specific parameters
    """
    try:
        # Get file paths
        files = DEFAULT_FILES[day]
        states_file = files['states']
        behavior_file = files['behavior']
        
        # Get clustering algorithm
        algorithm = ClusteringFactory.get_algorithm(algorithm_id, **algorithm_params)
        logging.info(f"Using {algorithm.get_name()} clustering algorithm")
        
        # Load data
        states_df, behavior_df = load_data(states_file, behavior_file)
        
        # Preprocess data
        X, timestamps, neuron_cols = preprocess_data(states_df)
        
        # Determine parameters and perform clustering
        params = algorithm.determine_parameters(X)
        logging.info(f"Using parameters: {params}")
        labels = algorithm.fit_predict(X, **params)
        
        # Visualize results
        visualize_clusters(X, labels, timestamps, algorithm.get_name(), 
                         f"{day} {algorithm.get_name()} Clustering Results")
        
        # Analyze behavior distribution
        analyze_cluster_behaviors(labels, timestamps, behavior_df, algorithm.get_name())
        
    except Exception as e:
        logging.error(f"Error in clustering analysis process: {e}")
        raise

if __name__ == "__main__":
    # Example usage:
    # KMeans clustering
    main(day='day6', algorithm_id=1, default_n_clusters=6)
    
    # DBSCAN clustering
    # main(day='day6', algorithm_id=2, k=4, default_eps=0.5)