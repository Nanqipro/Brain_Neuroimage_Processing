"""
Topology Clustering Analysis Module

This module performs clustering analysis on topology matrices to identify
temporal patterns in neuron connectivity. It supports multiple clustering algorithms
and provides visualization tools for analysis.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
from typing import Tuple, List, Optional, Dict, Any, Union
from abc import ABC, abstractmethod
import logging
import os
from scipy.cluster.hierarchy import dendrogram
from datetime import datetime

# File path configuration
DATA_DIR = '../datasets'  # Data directory path
RESULT_DIR = '../result'  # Results directory path
TOPOLOGY_FILE = os.path.join(DATA_DIR, 'Day9_topology_matrix_plus.xlsx')  # Topology matrix file
BEHAVIOR_FILE = os.path.join(DATA_DIR, 'Day9_with_behavior_labels_filled.xlsx')  # Behavior labels file

def setup_logging(algorithm_names: Union[str, List[str]]) -> str:
    """
    Set up logging configuration, create result directory and return log file path.

    Args:
        algorithm_names: Algorithm names to use (single string or list)

    Returns:
        str: Log file path
    """
    # Ensure result directory exists
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    # Generate log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if isinstance(algorithm_names, list):
        alg_str = "_".join(algorithm_names)
    else:
        alg_str = algorithm_names
    log_file = os.path.join(RESULT_DIR, f"clustering_{alg_str}_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return log_file

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
            max_k: Maximum number of clusters to test in parameter selection visualization
            default_n_clusters: Number of clusters to use in actual clustering
        """
        self.max_k = max_k
        self.default_n_clusters = default_n_clusters
    
    def determine_parameters(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Visualize parameter selection process but return manually specified n_clusters.
        This method will show the elbow curve and silhouette scores for reference,
        but won't automatically select the optimal k.
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
        
        # Plot parameter selection graphs for reference
        self._plot_parameter_selection(K_range, inertia, silhouette)
        
        # Calculate but don't use the optimal k
        optimal_k = K_range[np.argmax(silhouette)]
        logging.info(f"Reference: Optimal k based on silhouette score would be {optimal_k}")
        logging.info(f"Using manually specified n_clusters = {self.default_n_clusters}")
        
        # Return the manually specified n_clusters
        return {"n_clusters": self.default_n_clusters}
    
    def fit_predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform clustering with specified number of clusters.
        """
        n_clusters = kwargs.get("n_clusters", self.default_n_clusters)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        return kmeans.fit_predict(X)
    
    def get_name(self) -> str:
        return "KMeans"
    
    def _plot_parameter_selection(self, K_range, inertia, silhouette):
        """
        Plot elbow curve and silhouette scores for reference.
        """
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
        
        plt.title('KMeans Parameter Selection Reference')
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
        plt.xlabel('Points Sorted by Distance')
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

class AgglomerativeClusterer(ClusteringAlgorithm):
    """Agglomerative Hierarchical clustering implementation."""
    
    def __init__(self, max_k: int = 10, default_n_clusters: int = 6, linkage: str = 'ward'):
        """
        Initialize Agglomerative clusterer.

        Args:
            max_k: Maximum number of clusters to test
            default_n_clusters: Default number of clusters
            linkage: Linkage criterion to use ('ward', 'complete', 'average', 'single')
        """
        self.max_k = max_k
        self.default_n_clusters = default_n_clusters
        self.linkage = linkage
    
    def determine_parameters(self, X: np.ndarray) -> Dict[str, Any]:
        """Determine optimal parameters using silhouette scores."""
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
        
        # Plot dendrogram for reference
        self._plot_dendrogram(X)
        
        return {"n_clusters": self.default_n_clusters, "linkage": self.linkage}
    
    def _plot_dendrogram(self, X: np.ndarray):
        """Plot dendrogram of hierarchical clustering."""
        clustering = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage=self.linkage)
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
        return clustering.fit_predict(X)
    
    def get_name(self) -> str:
        return "Agglomerative"

class SpectralClusterer(ClusteringAlgorithm):
    """Spectral clustering implementation."""
    
    def __init__(self, max_k: int = 10, default_n_clusters: int = 6, gamma: float = 1.0):
        """
        Initialize Spectral clusterer.

        Args:
            max_k: Maximum number of clusters to test
            default_n_clusters: Default number of clusters
            gamma: Kernel coefficient for RBF kernel
        """
        self.max_k = max_k
        self.default_n_clusters = default_n_clusters
        self.gamma = gamma
    
    def determine_parameters(self, X: np.ndarray) -> Dict[str, Any]:
        """Determine optimal parameters using silhouette scores."""
        silhouette = []
        K_range = range(2, self.max_k + 1)
        
        # First, find optimal gamma if not specified
        if self.gamma == 1.0:
            # Calculate median of pairwise distances as a good default for gamma
            from sklearn.metrics.pairwise import pairwise_distances
            distances = pairwise_distances(X)
            self.gamma = 1.0 / (2 * np.median(distances) ** 2)
            logging.info(f"Automatically determined gamma: {self.gamma}")
        
        for k in K_range:
            clustering = SpectralClustering(
                n_clusters=k,
                affinity='rbf',
                gamma=self.gamma,
                assign_labels='kmeans',
                random_state=42
            )
            labels = clustering.fit_predict(X)
            score = silhouette_score(X, labels)
            silhouette.append(score)
        
        # Plot silhouette scores
        plt.figure(figsize=(10, 5))
        plt.plot(K_range, silhouette, 'o-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.title('Spectral Clustering: Silhouette Score vs Number of Clusters')
        plt.show()
        
        # Find optimal number of clusters for reference
        optimal_k = K_range[np.argmax(silhouette)]
        logging.info(f"Reference: Optimal number of clusters based on silhouette score would be {optimal_k}")
        logging.info(f"Using manually specified n_clusters = {self.default_n_clusters}")
        
        return {
            "n_clusters": self.default_n_clusters,  # Use manually specified value
            "affinity": 'rbf',
            "gamma": self.gamma,
            "assign_labels": 'kmeans'
        }
    
    def fit_predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        n_clusters = kwargs.get("n_clusters", self.default_n_clusters)
        affinity = kwargs.get("affinity", 'rbf')
        gamma = kwargs.get("gamma", self.gamma)
        assign_labels = kwargs.get("assign_labels", 'kmeans')
        
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity=affinity,
            gamma=gamma,
            assign_labels=assign_labels,
            random_state=42
        )
        return clustering.fit_predict(X)
    
    def get_name(self) -> str:
        return "Spectral"

class GMMClusterer(ClusteringAlgorithm):
    """Gaussian Mixture Model clustering implementation."""
    
    def __init__(self, max_k: int = 10, default_n_components: int = 6):
        """
        Initialize GMM clusterer.

        Args:
            max_k: Maximum number of components to test
            default_n_components: Default number of components to use
        """
        self.max_k = max_k
        self.default_n_components = default_n_components
    
    def determine_parameters(self, X: np.ndarray) -> Dict[str, Any]:
        """Determine optimal parameters using BIC scores."""
        bic = []
        K_range = range(2, self.max_k + 1)
        
        for k in K_range:
            gmm = GaussianMixture(n_components=k, random_state=42)
            gmm.fit(X)
            bic.append(gmm.bic(X))
        
        # Plot BIC scores
        plt.figure(figsize=(10, 5))
        plt.plot(K_range, bic, 'o-')
        plt.xlabel('Number of Components (k)')
        plt.ylabel('BIC Score')
        plt.title('GMM: BIC Score vs Number of Components')
        plt.show()
        
        # Find optimal number of components for reference
        optimal_k = K_range[np.argmin(bic)]  # Lower BIC is better
        logging.info(f"Reference: Optimal number of components based on BIC score would be {optimal_k}")
        logging.info(f"Using manually specified n_components = {self.default_n_components}")
        
        return {"n_components": self.default_n_components}
    
    def fit_predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        n_components = kwargs.get("n_components", self.default_n_components)
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        return gmm.fit_predict(X)
    
    def get_name(self) -> str:
        return "GMM"

class ClusteringFactory:
    """Factory class for creating clustering algorithm instances."""
    
    @staticmethod
    def get_algorithm(algorithm_id: int, **kwargs) -> ClusteringAlgorithm:
        """
        Get clustering algorithm instance.

        Args:
            algorithm_id: ID of the algorithm 
                (1: KMeans, 2: DBSCAN, 3: Agglomerative, 4: Spectral, 5: GMM)
            **kwargs: Algorithm-specific parameters
        """
        algorithms = {
            1: KMeansClusterer(
                max_k=kwargs.get('max_k', 10),
                default_n_clusters=kwargs.get('default_n_clusters', 4)
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
            4: SpectralClusterer(
                max_k=kwargs.get('max_k', 10),
                default_n_clusters=kwargs.get('default_n_clusters', 6),
                gamma=kwargs.get('gamma', 1.0)
            ),
            5: GMMClusterer(
                max_k=kwargs.get('max_k', 10),
                default_n_components=kwargs.get('default_n_components', 3)
            )
        }
        
        if algorithm_id not in algorithms:
            raise ValueError(f"Algorithm ID {algorithm_id} not supported")
        
        return algorithms[algorithm_id]

def load_and_preprocess_data(file_path: str = TOPOLOGY_FILE, 
                           behavior_file_path: str = BEHAVIOR_FILE) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load and preprocess topology matrix data.

    Args:
        file_path: Path to the Excel file containing topology matrix
        behavior_file_path: Path to the behavior labels file

    Returns:
        Tuple containing:
        - Preprocessed feature matrix
        - Array of timestamps
        - List of feature column names

    Raises:
        FileNotFoundError: If the specified files don't exist
        ValueError: If data is invalid or contains unexpected values
    """
    try:
        # Load topology data
        df = pd.read_excel(file_path)
        logging.info(f"Loaded data from {file_path}")
        
        # Separate timestamps and features
        timestamp_col = 'Time_Stamp'
        behavior_col = 'behavior'  # 排除behavior列
        feature_cols = [col for col in df.columns if col not in [timestamp_col, behavior_col]]
        
        # 确保数据为数值类型
        X = df[feature_cols].astype(float).values
        timestamps = df[timestamp_col].values
        
        # 检查并处理缺失值
        if pd.isna(X).any():
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
    algorithm_name: str
) -> None:
    """
    Visualize clustering results using PCA dimensionality reduction.

    Args:
        X: Preprocessed feature matrix
        labels: Cluster labels
        timestamps: Array of timestamps
        feature_cols: List of feature column names
        algorithm_name: Name of the clustering algorithm
    """
    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X)
    
    viz_df = pd.DataFrame({
        'PCA1': X_pca[:, 0],
        'PCA2': X_pca[:, 1],
        'Cluster': labels,
        'Time_Stamp': timestamps
    })
    
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
    plt.title(f'{algorithm_name} Clustering Results (PCA)')
    plt.legend(title='Clusters')
    plt.tight_layout()
    plt.show()

def analyze_clusters(
    labels: np.ndarray, 
    timestamps: np.ndarray, 
    algorithm_name: str, 
    behavior_file_path: str = BEHAVIOR_FILE,
    include_exp: bool = False
) -> None:
    """
    Analyze and print the timestamps belonging to each cluster, and match behavior labels.
    Can include or exclude 'Exp' behavior from distribution calculation.

    Args:
        labels: Cluster labels
        timestamps: Array of timestamps
        algorithm_name: Name of the clustering algorithm
        behavior_file_path: Path to the behavior labels file
        include_exp: Whether to include 'Exp' behavior in distribution calculation
    """
    # Load behavior data
    behavior_df = pd.read_excel(behavior_file_path)
    
    # Get all unique behaviors and show them at the start
    all_behaviors = sorted(behavior_df['behavior'].unique())
    logging.info("\nAll behavior types in dataset:")
    for behavior in all_behaviors:
        behavior_count = len(behavior_df[behavior_df['behavior'] == behavior])
        logging.info(f"- {behavior}: {behavior_count} total occurrences")
    logging.info("\nStarting cluster analysis...")
    
    unique_labels = np.unique(labels)
    for label in unique_labels:
        if algorithm_name == 'DBSCAN' and label == -1:
            cluster_name = "Noise points"
        else:
            cluster_name = f"{algorithm_name} Cluster {label}"
            
        # Get timestamps for this cluster and sort them
        cluster_timestamps = np.unique(timestamps[labels == label])
        cluster_timestamps.sort()
        
        # Get behavior labels for this cluster
        cluster_behaviors_df = behavior_df[behavior_df['stamp'].isin(cluster_timestamps)]
        
        if include_exp:
            # Include all behaviors in distribution
            behaviors_count = cluster_behaviors_df['behavior'].value_counts()
            total_count = len(cluster_behaviors_df)
        else:
            # Exclude 'Exp' from distribution calculation
            non_exp_df = cluster_behaviors_df[cluster_behaviors_df['behavior'] != 'Exp']
            behaviors_count = non_exp_df['behavior'].value_counts()
            total_count = len(non_exp_df)
        
        # Calculate percentages
        if total_count > 0:
            behavior_percentages = (behaviors_count / total_count * 100).round(2)
        else:
            behavior_percentages = behaviors_count  # Empty series if no behaviors
        
        # Output cluster information
        logging.info(f"\n{cluster_name}:")
        logging.info(f"Behavior distribution ({'including' if include_exp else 'excluding'} Exp):")
        
        # Show only behaviors that appear in this cluster
        for behavior, percentage in behavior_percentages.items():
            count = behaviors_count[behavior]
            logging.info(f"- {behavior}: {percentage}% ({count} occurrences)")
        
        # Log total number of timestamps and Exp information
        total_timestamps = len(cluster_timestamps)
        exp_count = len(cluster_behaviors_df[cluster_behaviors_df['behavior'] == 'Exp'])
        logging.info(f"\nTotal timestamps in cluster: {total_timestamps}")
        if not include_exp:
            logging.info(f"Including {exp_count} Exp timestamps (not included in distribution)")
        
        logging.info("\nTimestamps:")
        # Format timestamp output, 10 per line
        for i in range(0, len(cluster_timestamps), 10):
            batch = cluster_timestamps[i:i+10]
            logging.info(', '.join(map(str, batch)))

def main(
    file_path: str = TOPOLOGY_FILE,
    behavior_file_path: str = BEHAVIOR_FILE,
    algorithm_ids: Union[int, List[int]] = 3,
    include_exp: bool = False,
    **algorithm_params
) -> None:
    """
    Execute the complete clustering analysis workflow.

    Args:
        file_path: Path to topology matrix Excel file
        behavior_file_path: Path to behavior labels Excel file
        algorithm_ids: ID(s) of clustering algorithm(s) to use (single ID or list)
            1: KMeans
            2: DBSCAN
            3: Agglomerative
            4: Spectral
            5: GMM
        include_exp: Whether to include 'Exp' behavior in distribution calculation
        **algorithm_params: Algorithm-specific parameters
            For KMeans: 
                - max_k: Maximum number of clusters to test (default: 10)
                - default_n_clusters: Number of clusters to use (default: 4)
            For DBSCAN:
                - k: Number of neighbors (default: 4)
                - default_eps: Epsilon value (default: 0.5)
            For Agglomerative:
                - max_k: Maximum number of clusters to test
                - default_n_clusters: Default number of clusters
                - linkage: Linkage criterion ('ward', 'complete', 'average', 'single')
            For Spectral:
                - max_k: Maximum number of clusters to test
                - default_n_clusters: Default number of clusters
                - gamma: Kernel coefficient for RBF kernel
            For GMM:
                - max_k: Maximum number of components to test
                - default_n_components: Default number of components
    """
    try:
        # Convert single algorithm ID to list
        if isinstance(algorithm_ids, int):
            algorithm_ids = [algorithm_ids]
        
        # Get all selected algorithm instances
        algorithms = [ClusteringFactory.get_algorithm(aid, **algorithm_params) for aid in algorithm_ids]
        algorithm_names = [alg.get_name() for alg in algorithms]
        
        # Set up logging
        log_file = setup_logging(algorithm_names)
        logging.info(f"Starting clustering analysis, results will be saved to: {log_file}")
        logging.info(f"Using algorithms: {', '.join(algorithm_names)}")
        logging.info(f"Behavior distribution calculation: {'including' if include_exp else 'excluding'} Exp")
        
        # Load and preprocess data
        X_scaled, timestamps, feature_cols = load_and_preprocess_data(file_path, behavior_file_path)
        
        # Execute clustering analysis for each algorithm
        for algorithm in algorithms:
            logging.info(f"\n{'='*50}")
            logging.info(f"Executing {algorithm.get_name()} clustering algorithm")
            logging.info(f"{'='*50}")
            
            # Show parameter selection visualization and get parameters
            params = algorithm.determine_parameters(X_scaled)
            logging.info(f"Using parameters: {params}")
            
            # Perform clustering
            labels = algorithm.fit_predict(X_scaled, **params)
            
            # Visualize results
            visualize_clusters(X_scaled, labels, timestamps, feature_cols, algorithm.get_name())
            
            # Analyze clustering results and match behaviors
            analyze_clusters(labels, timestamps, algorithm.get_name(), behavior_file_path, include_exp)
        
        logging.info("\nClustering analysis complete!")
        
    except Exception as e:
        logging.error(f"Error in clustering analysis: {e}")
        raise

if __name__ == "__main__":
    # Example: Run multiple clustering algorithms
    main(algorithm_ids=[1], include_exp=True)  # Run KMeans, excluding Exp from distribution
    # Or include Exp in distribution
    # main(algorithm_ids=[1], include_exp=True)  # Run KMeans, including Exp in distribution