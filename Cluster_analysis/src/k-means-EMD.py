#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
K-means clustering with Earth Mover's Distance (EMD) for neuron calcium metrics data.
This script implements a custom K-means clustering algorithm using EMD as the distance metric,
and determines the optimal number of clusters using elbow and silhouette methods.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from numba import njit, prange

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
    'Start Time': 0.0,
    'Amplitude': 1.0,
    'Peak': 1.0,
    'Decay Time': 1.0,
    'Rise Time': 1.0,
    'Latency': 0.8,
    'Frequency': 0.8
}

def load_and_preprocess_data(file_path: str, sheet_name: str) -> pd.DataFrame:
    """
    Load and preprocess the data from Excel file.
    
    Args:
        file_path: Path to the Excel file
        sheet_name: Name of the sheet to read
        
    Returns:
        Preprocessed DataFrame with weighted features
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    if df[FEATURES].isnull().values.any():
        print("Warning: Data contains missing values, removing rows with NaN...")
        df = df.dropna(subset=FEATURES).reset_index(drop=True)
    
    X = df[FEATURES].values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    weight_values = np.array([FEATURE_WEIGHTS[feature] for feature in FEATURES])
    X_weighted = X_scaled * weight_values
    
    return df, X_weighted

@njit
def emd_distance(u_values: np.ndarray, v_values: np.ndarray) -> float:
    """
    Calculate Earth Mover's Distance between two vectors.
    
    Args:
        u_values: First vector
        v_values: Second vector
        
    Returns:
        EMD distance value
    """
    u_sorted = np.sort(u_values)
    v_sorted = np.sort(v_values)
    return np.sum(np.abs(np.cumsum(u_sorted - v_sorted)))

def k_means_emd(X: np.ndarray, n_clusters: int, max_iters: int = 100, 
                random_state: int = None) -> np.ndarray:
    """
    Perform K-means clustering using EMD distance metric.
    
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
        distance_matrix = np.zeros((n_samples, n_clusters))
        for i in prange(n_samples):
            for j in range(n_clusters):
                distance_matrix[i, j] = emd_distance(X[i], centroids[j])

        new_labels = np.argmin(distance_matrix, axis=1)

        if np.array_equal(new_labels, labels):
            print(f"Algorithm converged after {iteration+1} iterations.")
            break
            
        labels = new_labels

        for k in range(n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                centroids[k] = np.mean(cluster_points, axis=0)

    return labels

@njit
def compute_wcsd(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    """
    Compute Within-Cluster Sum of Distances.
    
    Args:
        X: Input data matrix
        labels: Cluster labels
        centroids: Cluster centroids
        
    Returns:
        Total within-cluster distance
    """
    total_distance = 0.0
    for k in prange(len(centroids)):
        cluster_points = X[labels == k]
        centroid = centroids[k]
        for i in range(cluster_points.shape[0]):
            total_distance += emd_distance(cluster_points[i], centroid)
    return total_distance

def plot_elbow_curve(K: range, wcsd: list):
    """Plot elbow curve for determining optimal number of clusters."""
    plt.figure(figsize=(8, 4))
    plt.plot(K, wcsd, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Within-Cluster Sum of Distances (WCSD)')
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
        wcsd = []
        K = range(1, 6)
        for k in K:
            print(f"Computing clustering for k={k}...")
            labels = k_means_emd(X_weighted, n_clusters=k, random_state=0)
            centroids = np.array([X_weighted[labels == i].mean(axis=0) for i in range(k)])
            wcsd.append(compute_wcsd(X_weighted, labels, centroids))
        plot_elbow_curve(K, wcsd)
        
        # Calculate and plot silhouette curve
        silhouette_scores = []
        K = range(2, 6)
        for k in K:
            print(f"Computing silhouette score for k={k}...")
            labels = k_means_emd(X_weighted, n_clusters=k, random_state=0)
            sample_indices = np.random.choice(len(X_weighted), 
                                           size=min(100, len(X_weighted)), 
                                           replace=False)
            sample_X = X_weighted[sample_indices]
            sample_labels = labels[sample_indices]
            
            distance_matrix = np.zeros((len(sample_X), len(sample_X)))
            for i in prange(len(sample_X)):
                for j in range(i + 1, len(sample_X)):
                    dist = emd_distance(sample_X[i], sample_X[j])
                    distance_matrix[i, j] = dist
                    distance_matrix[j, i] = dist
                    
            silhouette_avg = silhouette_score(distance_matrix, sample_labels, 
                                            metric='precomputed')
            silhouette_scores.append(silhouette_avg)
        plot_silhouette_curve(K, silhouette_scores)
        
        # Perform final clustering with optimal k
        final_labels = k_means_emd(X_weighted, n_clusters=optimal_k, random_state=0)
        metrics_df['k-means-EMD'] = final_labels
        
        # Save results
        metrics_df.to_excel(input_file, index=False, sheet_name=sheet_name)
        print(f'Clustering results saved to: {input_file}')
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()
