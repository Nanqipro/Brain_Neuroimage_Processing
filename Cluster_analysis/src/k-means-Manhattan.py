#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
K-means clustering with Manhattan distance for neuron calcium metrics data.
This script implements an optimized K-means clustering algorithm using Manhattan distance
and determines the optimal number of clusters using elbow and silhouette methods.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from numba import njit

# Suppress warnings
warnings.filterwarnings("ignore")

# Constants
FEATURES = [
    'Start Time',
    'Amplitude',
    'Peak',
    'Decay Time',
    'Rise Time',
    'Latency',
    'Frequency'
]

FEATURE_WEIGHTS = {
    'Start Time': 0.1,
    'Amplitude': 2.0,
    'Peak': 2.0,
    'Decay Time': 1.0,
    'Rise Time': 1.0,
    'Latency': 1.5,
    'Frequency': 1.5
}

def load_and_preprocess_data(file_path: str, sheet_name: str) -> tuple:
    """
    Load and preprocess the data from Excel file.
    
    Args:
        file_path: Path to the Excel file
        sheet_name: Name of the sheet to read
        
    Returns:
        Tuple of (DataFrame, weighted and scaled features array)
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    if df[FEATURES].isnull().values.any():
        print("Warning: Data contains missing values, removing rows with NaN...")
        df = df.dropna(subset=FEATURES).reset_index(drop=True)
    
    X = df[FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    weight_values = np.array([FEATURE_WEIGHTS[feature] for feature in FEATURES])
    X_weighted = X_scaled * weight_values
    
    return df, X_weighted

def k_means_manhattan(X: np.ndarray, n_clusters: int, max_iters: int = 100, 
                     random_state: int = None) -> np.ndarray:
    """
    Perform K-means clustering using Manhattan distance.
    
    Args:
        X: Input data matrix
        n_clusters: Number of clusters
        max_iters: Maximum number of iterations
        random_state: Random seed for reproducibility
        
    Returns:
        Array of cluster labels
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples, n_features = X.shape
    initial_centroids_indices = np.random.choice(n_samples, n_clusters, replace=False)
    centroids = X[initial_centroids_indices].copy()
    labels = np.full(n_samples, -1)

    for iteration in range(max_iters):
        # Calculate Manhattan distances using cityblock metric
        distance_matrix = cdist(X, centroids, metric='cityblock')
        new_labels = np.argmin(distance_matrix, axis=1)

        if np.array_equal(new_labels, labels):
            print(f"Algorithm converged after {iteration+1} iterations.")
            break
            
        labels = new_labels

        # Update centroids using median
        for k in range(n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                centroids[k] = np.median(cluster_points, axis=0)

    return labels

@njit
def compute_wcss(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    """
    Compute Within-Cluster Sum of Distances using Manhattan distance.
    
    Args:
        X: Input data matrix
        labels: Cluster labels
        centroids: Cluster centroids
        
    Returns:
        Total within-cluster sum of distances
    """
    total_distance = 0.0
    for i in range(len(centroids)):
        cluster_points = X[labels == i]
        centroid = centroids[i]
        total_distance += np.sum(np.abs(cluster_points - centroid))
    return total_distance

def plot_elbow_curve(K: range, wcss: list):
    """Plot elbow curve for determining optimal number of clusters."""
    plt.figure(figsize=(8, 4))
    plt.plot(K, wcss, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Within-Cluster Sum of Distances')
    plt.title('Elbow Method for Optimal k')
    plt.xticks(K)
    plt.show()

def plot_silhouette_curve(K: range, scores: list):
    """Plot silhouette curve for determining optimal number of clusters."""
    plt.figure(figsize=(8, 4))
    plt.plot(K, scores, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Average Silhouette Score')
    plt.title('Silhouette Method for Optimal k')
    plt.xticks(K)
    plt.show()

def main():
    """Main execution function."""
    # File paths and parameters
    input_file = '../datasets/Day3_Neuron_Calcium_Metrics.xlsx'
    sheet_name = 'Windows100_step10'
    optimal_k = 6
    
    try:
        # Load and preprocess data
        metrics_df, X_weighted = load_and_preprocess_data(input_file, sheet_name)
        
        # Calculate and plot elbow curve
        wcss = []
        K = range(1, 11)
        for k in K:
            print(f"Computing clustering for k={k}...")
            labels = k_means_manhattan(X_weighted, n_clusters=k, random_state=0)
            centroids = np.array([np.median(X_weighted[labels == i], axis=0) 
                                for i in range(k)])
            wcss.append(compute_wcss(X_weighted, labels, centroids))
        plot_elbow_curve(K, wcss)
        
        # Calculate and plot silhouette curve
        silhouette_scores = []
        K = range(2, 11)
        for k in K:
            print(f"Computing silhouette score for k={k}...")
            labels = k_means_manhattan(X_weighted, n_clusters=k, random_state=0)
            silhouette_avg = silhouette_score(X_weighted, labels, metric='manhattan')
            silhouette_scores.append(silhouette_avg)
        plot_silhouette_curve(K, silhouette_scores)
        
        # Perform final clustering with optimal k
        final_labels = k_means_manhattan(X_weighted, n_clusters=optimal_k, random_state=0)
        metrics_df['k-means-Manhattan'] = final_labels
        
        # Save results
        metrics_df.to_excel(input_file, index=False, sheet_name=sheet_name)
        print(f'Clustering results saved to: {input_file}')
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()
