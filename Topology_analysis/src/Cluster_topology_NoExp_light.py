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
    
    def __init__(self, max_k: int = 10, default_n_clusters: int = 5):
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
        
        # Plot parameter selection visualization
        self._plot_parameter_selection(K_range, inertia, silhouette)
        
        # Find optimal k using silhouette score
        optimal_k = K_range[np.argmax(silhouette)]
        logging.info(f"Optimal number of clusters based on silhouette score: {optimal_k}")
        
        return {"n_clusters": self.default_n_clusters}
    
    def fit_predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        n_clusters = kwargs.get("n_clusters", self.default_n_clusters)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        logging.info(f"KMeans clustering completed, silhouette score: {score:.3f}")
        return labels
    
    def get_name(self) -> str:
        return "KMeans"
    
    def _plot_parameter_selection(self, K_range, inertia, silhouette):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot inertia (elbow curve)
        ax1.plot(K_range, inertia, 'bo-')
        ax1.set_xlabel('Number of Clusters (K)')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Curve')
        
        # Plot silhouette scores
        ax2.plot(K_range, silhouette, 'ro-')
        ax2.set_xlabel('Number of Clusters (K)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Score vs K')
        
        plt.tight_layout()
        plt.show()

class DBSCANClusterer(ClusteringAlgorithm):
    """DBSCAN clustering implementation."""
    
    def __init__(self, k: int = 4, default_eps: float = 0.5):
        self.k = k
        self.default_eps = default_eps
    
    def determine_parameters(self, X: np.ndarray) -> Dict[str, Any]:
        # Use k-distance graph to determine eps
        neighbors = NearestNeighbors(n_neighbors=self.k)
        neighbors_fit = neighbors.fit(X)
        distances, _ = neighbors_fit.kneighbors(X)
        distances = np.sort(distances[:, self.k-1])
        
        # Plot k-distance graph
        plt.figure(figsize=(10, 5))
        plt.plot(distances)
        plt.axhline(y=np.percentile(distances, 90), color='r', linestyle='--')
        plt.ylabel(f'{self.k}-distance')
        plt.xlabel('Points Sorted by Distance')
        plt.title(f'k-distance Graph (k={self.k}) for DBSCAN')
        plt.show()
        
        # Use 90th percentile of distances as eps
        eps = np.percentile(distances, 90)
        logging.info(f"Suggested eps value: {eps:.3f}")
        
        return {
            "eps": eps,
            "min_samples": self.k + 1
        }
    
    def fit_predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
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
        algorithms = {
            1: KMeansClusterer(
                max_k=kwargs.get('max_k', 10),
                default_n_clusters=kwargs.get('default_n_clusters', 5)
            ),
            2: DBSCANClusterer(
                k=kwargs.get('k', 4),
                default_eps=kwargs.get('default_eps', 0.5)
            )
        }
        
        if algorithm_id not in algorithms:
            raise ValueError(f"Algorithm ID {algorithm_id} not supported")
        
        return algorithms[algorithm_id]

def load_and_preprocess_data(file_path: str, behavior_file_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load and preprocess neuron state data."""
    try:
        # Load data
        states_df = pd.read_excel(file_path)
        behavior_df = pd.read_excel(behavior_file_path)
        logging.info(f"Loaded data from {file_path}")
        
        # Filter out Exp behavior if present
        if 'behavior' in behavior_df.columns:
            non_exp_mask = behavior_df['behavior'] != 'Exp'
            behavior_df = behavior_df[non_exp_mask]
            states_df = states_df[states_df['Time_Stamp'].isin(behavior_df['stamp'])]
        
        # Separate timestamps and features
        timestamp_col = 'Time_Stamp'
        feature_cols = [col for col in states_df.columns if col != timestamp_col]
        
        X = states_df[feature_cols].values
        timestamps = states_df[timestamp_col].values
        
        # Handle missing values if any
        if np.isnan(X).any():
            logging.warning("Found missing values, filling with zeros")
            X = np.nan_to_num(X)
        
        logging.info(f"Data shape: {X.shape}")
        return X, timestamps, feature_cols
    
    except Exception as e:
        logging.error(f"Error in data preprocessing: {e}")
        raise

def visualize_clusters(X: np.ndarray, labels: np.ndarray, timestamps: np.ndarray,
                      feature_cols: List[str], algorithm_name: str) -> None:
    """Visualize clustering results using PCA."""
    # Reduce dimensionality for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Create visualization DataFrame
    viz_df = pd.DataFrame({
        'PCA1': X_pca[:, 0],
        'PCA2': X_pca[:, 1],
        'Cluster': labels,
        'Time_Stamp': timestamps
    })
    
    # Create interactive plot
    fig = px.scatter(
        viz_df,
        x='PCA1',
        y='PCA2',
        color='Cluster',
        title=f'{algorithm_name} Clustering Results',
        hover_data=['Time_Stamp']
    )
    fig.show()
    
    # Create static plot
    plt.figure(figsize=(12, 8))
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
            str(viz_df['Time_Stamp'].iloc[i]),
            fontsize=9
        )
    
    plt.title(f'{algorithm_name} Clustering Results')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.show()

def analyze_clusters(labels: np.ndarray, timestamps: np.ndarray, algorithm_name: str, behavior_file_path: str) -> None:
    """Analyze behavior distribution in each cluster."""
    behavior_df = pd.read_excel(behavior_file_path)
    
    unique_labels = np.unique(labels)
    for label in unique_labels:
        cluster_name = "Noise points" if algorithm_name == 'DBSCAN' and label == -1 else f"Cluster {label}"
        
        # Get cluster timestamps
        cluster_timestamps = timestamps[labels == label]
        
        # Get behavior distribution
        cluster_behaviors = behavior_df[behavior_df['stamp'].isin(cluster_timestamps)]['behavior']
        behavior_counts = cluster_behaviors.value_counts()
        behavior_percentages = (behavior_counts / len(cluster_timestamps) * 100).round(2)
        
        # Output results
        logging.info(f"\n{cluster_name}:")
        logging.info(f"Contains {len(cluster_timestamps)} timestamps")
        logging.info("\nBehavior distribution:")
        for behavior, percentage in behavior_percentages.items():
            logging.info(f"- {behavior}: {percentage}% ({behavior_counts[behavior]} occurrences)")
        
        logging.info("\nTimestamps:")
        timestamps_str = ', '.join(map(str, sorted(cluster_timestamps)))
        logging.info(timestamps_str)

def main(day: str = 'day6', algorithm_id: int = 1, **algorithm_params) -> None:
    """Execute complete clustering analysis workflow."""
    try:
        # Get file paths
        files = DEFAULT_FILES[day]
        states_file = files['states']
        behavior_file = files['behavior']
        
        # Get clustering algorithm
        algorithm = ClusteringFactory.get_algorithm(algorithm_id, **algorithm_params)
        logging.info(f"Using {algorithm.get_name()} clustering algorithm")
        
        # Load and preprocess data
        X, timestamps, feature_cols = load_and_preprocess_data(states_file, behavior_file)
        
        # Determine parameters and perform clustering
        params = algorithm.determine_parameters(X)
        logging.info(f"Using parameters: {params}")
        labels = algorithm.fit_predict(X, **params)
        
        # Visualize results
        visualize_clusters(X, labels, timestamps, feature_cols, algorithm.get_name())
        
        # Analyze behavior distribution
        analyze_clusters(labels, timestamps, algorithm.get_name(), behavior_file)
        
    except Exception as e:
        logging.error(f"Error in clustering analysis: {e}")
        raise

if __name__ == "__main__":
    # Example usage:
    # KMeans clustering
    main(day='day6', algorithm_id=1, default_n_clusters=4)
    
    # DBSCAN clustering
    # main(day='day6', algorithm_id=2, k=4, default_eps=0.5)