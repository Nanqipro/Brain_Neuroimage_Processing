"""
Neuron State Clustering Analysis Module

This module performs clustering analysis on neuron activation states to:
1. Identify similar neuron activation patterns
2. Group timestamps into different behavioral patterns
3. Analyze behavior label distribution for each cluster
"""

import pandas as pd
import numpy as np
from sklearn.cluster import (
    KMeans, DBSCAN, AgglomerativeClustering,
    MeanShift, estimate_bandwidth, AffinityPropagation
)
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
from scipy.cluster.hierarchy import dendrogram

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

class AgglomerativeClusterer(ClusteringAlgorithm):
    """Agglomerative Hierarchical clustering implementation."""
    
    def __init__(self, max_k: int = 10, default_n_clusters: int = 6, linkage: str = 'ward'):
        self.max_k = max_k
        self.default_n_clusters = default_n_clusters
        self.linkage = linkage
    
    def determine_parameters(self, X: np.ndarray) -> Dict[str, Any]:
        silhouette = []
        K_range = range(2, self.max_k + 1)
        
        for k in K_range:
            clustering = AgglomerativeClustering(n_clusters=k, linkage=self.linkage)
            labels = clustering.fit_predict(X)
            score = silhouette_score(X, labels)
            silhouette.append(score)
        
        # Plot silhouette scores
        plt.figure(figsize=(10, 5))
        plt.plot(K_range, silhouette, 'o-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score vs Number of Clusters')
        plt.show()
        
        # Plot dendrogram
        self._plot_dendrogram(X)
        
        optimal_k = K_range[np.argmax(silhouette)]
        logging.info(f"Optimal number of clusters based on silhouette score: {optimal_k}")
        
        return {"n_clusters": self.default_n_clusters, "linkage": self.linkage}
    
    def _plot_dendrogram(self, X: np.ndarray):
        clustering = AgglomerativeClustering(
            distance_threshold=0,
            n_clusters=None,
            linkage=self.linkage
        )
        clustering.fit(X)
        
        counts = np.zeros(clustering.children_.shape[0])
        n_samples = len(clustering.labels_)
        for i, merge in enumerate(clustering.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack([
            clustering.children_,
            clustering.distances_,
            counts
        ]).astype(float)

        plt.figure(figsize=(10, 7))
        dendrogram(linkage_matrix)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Sample index or (cluster size)')
        plt.ylabel('Distance')
        plt.show()
    
    def fit_predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        n_clusters = kwargs.get("n_clusters", self.default_n_clusters)
        linkage = kwargs.get("linkage", self.linkage)
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        labels = clustering.fit_predict(X)
        score = silhouette_score(X, labels)
        logging.info(f"Agglomerative clustering completed, silhouette score: {score:.3f}")
        return labels
    
    def get_name(self) -> str:
        return "Agglomerative"

class MeanShiftClusterer(ClusteringAlgorithm):
    """MeanShift clustering implementation."""
    
    def __init__(self, quantile: float = 0.3, n_samples: int = 100):
        self.quantile = quantile
        self.n_samples = n_samples
    
    def determine_parameters(self, X: np.ndarray) -> Dict[str, Any]:
        # Estimate bandwidth using a subset of data
        bandwidth = estimate_bandwidth(
            X, 
            quantile=self.quantile,
            n_samples=min(self.n_samples, X.shape[0])
        )
        logging.info(f"Estimated bandwidth: {bandwidth:.3f}")
        
        # Test different quantiles
        quantiles = [0.1, 0.2, 0.3, 0.4, 0.5]
        bandwidths = []
        n_clusters = []
        silhouette_scores = []
        
        for q in quantiles:
            bw = estimate_bandwidth(X, quantile=q, n_samples=min(self.n_samples, X.shape[0]))
            ms = MeanShift(bandwidth=bw)
            labels = ms.fit_predict(X)
            
            bandwidths.append(bw)
            n_clusters.append(len(np.unique(labels)))
            if len(np.unique(labels)) > 1:
                silhouette_scores.append(silhouette_score(X, labels))
            else:
                silhouette_scores.append(0)
        
        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(quantiles, bandwidths, 'o-')
        ax1.set_xlabel('Quantile')
        ax1.set_ylabel('Bandwidth')
        ax1.set_title('Bandwidth vs Quantile')
        
        ax2.plot(quantiles, n_clusters, 'o-')
        ax2.set_xlabel('Quantile')
        ax2.set_ylabel('Number of Clusters')
        ax2.set_title('Number of Clusters vs Quantile')
        
        plt.tight_layout()
        plt.show()
        
        # Plot silhouette scores
        plt.figure(figsize=(10, 5))
        plt.plot(quantiles, silhouette_scores, 'o-')
        plt.xlabel('Quantile')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score vs Quantile')
        plt.show()
        
        return {"bandwidth": bandwidth}
    
    def fit_predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        bandwidth = kwargs.get("bandwidth")
        ms = MeanShift(bandwidth=bandwidth)
        labels = ms.fit_predict(X)
        n_clusters = len(np.unique(labels))
        logging.info(f"MeanShift found {n_clusters} clusters")
        
        if n_clusters > 1:
            score = silhouette_score(X, labels)
            logging.info(f"Silhouette score: {score:.3f}")
        
        return labels
    
    def get_name(self) -> str:
        return "MeanShift"

class AffinityPropagationClusterer(ClusteringAlgorithm):
    """Affinity Propagation clustering implementation."""
    
    def __init__(self, damping: float = 0.5, preference: float = None):
        self.damping = damping
        self.preference = preference
    
    def determine_parameters(self, X: np.ndarray) -> Dict[str, Any]:
        # Test different damping values
        damping_values = [0.5, 0.6, 0.7, 0.8, 0.9]
        n_clusters = []
        silhouette_scores = []
        
        for damping in damping_values:
            af = AffinityPropagation(damping=damping, random_state=42)
            labels = af.fit_predict(X)
            
            n_clusters.append(len(np.unique(labels)))
            if len(np.unique(labels)) > 1:
                silhouette_scores.append(silhouette_score(X, labels))
            else:
                silhouette_scores.append(0)
        
        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(damping_values, n_clusters, 'o-')
        ax1.set_xlabel('Damping')
        ax1.set_ylabel('Number of Clusters')
        ax1.set_title('Number of Clusters vs Damping')
        
        ax2.plot(damping_values, silhouette_scores, 'o-')
        ax2.set_xlabel('Damping')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Score vs Damping')
        
        plt.tight_layout()
        plt.show()
        
        # Choose optimal damping
        optimal_idx = np.argmax(silhouette_scores)
        optimal_damping = damping_values[optimal_idx]
        logging.info(f"Optimal damping: {optimal_damping}")
        
        return {
            "damping": optimal_damping,
            "preference": self.preference
        }
    
    def fit_predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        damping = kwargs.get("damping", self.damping)
        preference = kwargs.get("preference", self.preference)
        
        af = AffinityPropagation(
            damping=damping,
            preference=preference,
            random_state=42
        )
        labels = af.fit_predict(X)
        n_clusters = len(np.unique(labels))
        logging.info(f"Affinity Propagation found {n_clusters} clusters")
        
        if n_clusters > 1:
            score = silhouette_score(X, labels)
            logging.info(f"Silhouette score: {score:.3f}")
        
        return labels
    
    def get_name(self) -> str:
        return "AffinityPropagation"

class ClusteringFactory:
    """Factory class for creating clustering algorithm instances."""
    
    @staticmethod
    def get_algorithm(algorithm_id: int, **kwargs) -> ClusteringAlgorithm:
        """
        Get clustering algorithm instance.

        Args:
            algorithm_id: Algorithm identifier 
                (1: KMeans, 2: DBSCAN, 3: Agglomerative, 4: MeanShift, 5: AffinityPropagation)
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
            ),
            3: AgglomerativeClusterer(
                max_k=kwargs.get('max_k', 10),
                default_n_clusters=kwargs.get('default_n_clusters', 6),
                linkage=kwargs.get('linkage', 'ward')
            ),
            4: MeanShiftClusterer(
                quantile=kwargs.get('quantile', 0.3),
                n_samples=kwargs.get('n_samples', 100)
            ),
            5: AffinityPropagationClusterer(
                damping=kwargs.get('damping', 0.5),
                preference=kwargs.get('preference', None)
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
        algorithm_id: Clustering algorithm ID 
            1: KMeans
            2: DBSCAN
            3: Agglomerative
            4: MeanShift
            5: AffinityPropagation
        **algorithm_params: Algorithm-specific parameters
            For KMeans:
                - max_k: Maximum number of clusters to test
                - default_n_clusters: Number of clusters to use
            For DBSCAN:
                - k: Number of neighbors
                - default_eps: Epsilon value
            For Agglomerative:
                - max_k: Maximum number of clusters to test
                - default_n_clusters: Number of clusters to use
                - linkage: Linkage criterion ('ward', 'complete', 'average', 'single')
            For MeanShift:
                - quantile: Quantile for bandwidth estimation
                - n_samples: Number of samples for bandwidth estimation
            For AffinityPropagation:
                - damping: Damping factor
                - preference: Preference parameter
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