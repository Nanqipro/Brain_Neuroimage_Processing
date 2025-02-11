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

class AgglomerativeClusterer(ClusteringAlgorithm):
    """Agglomerative Hierarchical clustering implementation."""
    
    def __init__(self, max_k: int = 10, default_n_clusters: int = 5, linkage: str = 'ward'):
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
        algorithms = {
            1: KMeansClusterer(
                max_k=kwargs.get('max_k', 10),
                default_n_clusters=kwargs.get('default_n_clusters', 5)
            ),
            2: DBSCANClusterer(
                k=kwargs.get('k', 4),
                default_eps=kwargs.get('default_eps', 0.5)
            ),
            3: AgglomerativeClusterer(
                max_k=kwargs.get('max_k', 10),
                default_n_clusters=kwargs.get('default_n_clusters', 5),
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
    """
    Execute complete clustering analysis workflow.
    
    Args:
        day: Which day's data to analyze ('day3', 'day6', or 'day9')
        algorithm_id: ID of the clustering algorithm to use
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