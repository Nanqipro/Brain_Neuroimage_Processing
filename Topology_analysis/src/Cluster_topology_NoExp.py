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
import os

# 文件路径配置
DATA_DIR = '../datasets'  # 数据文件夹路径
TOPOLOGY_FILE = os.path.join(DATA_DIR, 'Day6_topology_matrix_plus.xlsx')  # 拓扑矩阵文件
BEHAVIOR_FILE = os.path.join(DATA_DIR, 'Day6_with_behavior_labels_filled.xlsx')  # 行为标签文件

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

def load_and_preprocess_data(file_path: str = TOPOLOGY_FILE, 
                           behavior_file_path: str = BEHAVIOR_FILE) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load and preprocess topology matrix data, excluding rows with 'Exp' behavior.

    Args:
        file_path: Path to the Excel file containing topology matrix
        behavior_file_path: Path to the Excel file containing behavior labels

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
        # Load behavior data
        behavior_df = pd.read_excel(behavior_file_path)
        
        logging.info(f"Loaded data from {file_path} and {behavior_file_path}")
        
        # Filter out rows with 'Exp' behavior
        non_exp_mask = behavior_df['behavior'] != 'Exp'
        behavior_df = behavior_df[non_exp_mask]
        
        # Get corresponding timestamps from topology data
        df = df[df['Time_Stamp'].isin(behavior_df['stamp'])]
        
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
        
        logging.info(f"Data shape after excluding 'Exp' behavior: {X_scaled.shape}")
        
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
    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X)
    
    viz_df = pd.DataFrame({
        'PCA1': X_pca[:, 0],
        'PCA2': X_pca[:, 1],
        'Cluster': labels,
        'Time_Stamp': timestamps
    })
    
    title = f'{algorithm_name} Clustering Results (PCA)'
    
    if interactive:
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
        plt.title(title)
        plt.legend(title='Cluster')
        plt.tight_layout()
        plt.show()

def analyze_clusters(labels: np.ndarray, timestamps: np.ndarray, algorithm_name: str, behavior_file_path: str = BEHAVIOR_FILE) -> None:
    """
    Analyze and print the timestamps belonging to each cluster, and match behavior labels.

    Args:
        labels: Cluster labels
        timestamps: Array of timestamps
        algorithm_name: Name of the clustering algorithm
        behavior_file_path: Path to the behavior labels file
    """
    # 加载行为数据
    behavior_df = pd.read_excel(behavior_file_path)
    
    unique_labels = np.unique(labels)
    for label in unique_labels:
        if algorithm_name == 'DBSCAN' and label == -1:
            cluster_name = "Noise points"
        else:
            cluster_name = f"{algorithm_name} Cluster {label}"
            
        # 获取该簇的时间戳并去重排序
        cluster_timestamps = np.unique(timestamps[labels == label])
        # 确保时间戳是按照顺序排列的
        cluster_timestamps.sort()
        
        # 获取该簇的行为标签
        cluster_behaviors = behavior_df[behavior_df['stamp'].isin(cluster_timestamps)]['behavior'].value_counts()
        total_timestamps = len(cluster_timestamps)
        
        # 计算每个行为标签的百分比
        behavior_percentages = (cluster_behaviors / total_timestamps * 100).round(2)
        
        # 输出聚类信息
        logging.info(f"\n{cluster_name}:")
        logging.info("Behavior distribution:")
        for behavior, percentage in behavior_percentages.items():
            logging.info(f"- {behavior}: {percentage}% ({cluster_behaviors[behavior]} occurrences)")
        
        logging.info("\nTimestamps:")
        # 格式化输出时间戳，每行显示10个
        for i in range(0, len(cluster_timestamps), 10):
            batch = cluster_timestamps[i:i+10]
            logging.info(', '.join(map(str, batch)))

def main(
    file_path: str = TOPOLOGY_FILE,
    behavior_file_path: str = BEHAVIOR_FILE,
    algorithm_id: int = 1,
    **algorithm_params
) -> None:
    """

    Execute the complete clustering analysis workflow.

    Args:
        file_path: Path to the topology matrix Excel file
        behavior_file_path: Path to the behavior labels Excel file
        algorithm_id: ID of the clustering algorithm to use (1: KMeans, 2: DBSCAN)
        **algorithm_params: Algorithm-specific parameters
            For KMeans: 
                - max_k: Maximum number of clusters to test in visualization (default: 10)
                - default_n_clusters: Number of clusters to use (default: 6)
            For DBSCAN:
                - k: Number of neighbors (default: 4)
                - default_eps: Epsilon value (default: 0.5)
    """
    try:
        # Get the selected clustering algorithm with custom parameters
        algorithm = ClusteringFactory.get_algorithm(algorithm_id, **algorithm_params)
        logging.info(f"Using {algorithm.get_name()} clustering algorithm")
        
        # Load and preprocess data
        X_scaled, timestamps, feature_cols = load_and_preprocess_data(file_path, behavior_file_path)
        
        # Show parameter selection visualization and get specified parameters
        params = algorithm.determine_parameters(X_scaled)
        logging.info(f"Using parameters: {params}")
        
        # Perform clustering
        labels = algorithm.fit_predict(X_scaled, **params)
        
        # Visualize results
        visualize_clusters(X_scaled, labels, timestamps, feature_cols, 
                         algorithm.get_name(), interactive=True)
        visualize_clusters(X_scaled, labels, timestamps, feature_cols,
                         algorithm.get_name(), interactive=False)
        
        # Analyze clusters and match behaviors
        analyze_clusters(labels, timestamps, algorithm.get_name(), behavior_file_path)
        
    except Exception as e:
        logging.error(f"Error in clustering analysis: {e}")
        raise

if __name__ == "__main__":
    # 可以在这里修改文件路径
    main()