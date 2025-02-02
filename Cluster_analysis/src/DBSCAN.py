#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DBSCAN clustering for neuron calcium metrics data.
This script implements DBSCAN clustering algorithm and visualizes the results
using various metrics and plots.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

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

def perform_dbscan(X: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """
    Perform DBSCAN clustering.
    
    Args:
        X: Input data matrix
        eps: Maximum distance between samples for neighborhood
        min_samples: Minimum number of samples in a neighborhood
        
    Returns:
        Array of cluster labels
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    return labels

def evaluate_clustering(X: np.ndarray, labels: np.ndarray) -> tuple:
    """
    Evaluate clustering results.
    
    Args:
        X: Input data matrix
        labels: Cluster labels
        
    Returns:
        Tuple of (number of clusters, number of noise points, silhouette score)
    """
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    silhouette_avg = 0
    if n_clusters > 1:
        # Calculate silhouette score only for non-noise points
        mask = labels != -1
        if np.sum(mask) > 1:
            silhouette_avg = silhouette_score(X[mask], labels[mask])
    
    return n_clusters, n_noise, silhouette_avg

def plot_evaluation_metrics(eps_range: list, metrics: list):
    """Plot evaluation metrics for different epsilon values."""
    n_clusters_list, n_noise_list, silhouette_scores = zip(*metrics)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    
    ax1.plot(eps_range, n_clusters_list, 'bo-')
    ax1.set_xlabel('Epsilon')
    ax1.set_ylabel('Number of Clusters')
    ax1.set_title('Number of Clusters vs Epsilon')
    
    ax2.plot(eps_range, n_noise_list, 'ro-')
    ax2.set_xlabel('Epsilon')
    ax2.set_ylabel('Number of Noise Points')
    ax2.set_title('Number of Noise Points vs Epsilon')
    
    ax3.plot(eps_range, silhouette_scores, 'go-')
    ax3.set_xlabel('Epsilon')
    ax3.set_ylabel('Silhouette Score')
    ax3.set_title('Silhouette Score vs Epsilon')
    
    plt.tight_layout()
    plt.show()

def main():
    """Main execution function."""
    # File paths and parameters
    input_file = '../datasets/Day3_Neuron_Calcium_Metrics.xlsx'
    sheet_name = 'Windows100_step10'
    min_samples = 5
    optimal_eps = 0.5
    
    try:
        # Load and preprocess data
        metrics_df, X_weighted = load_and_preprocess_data(input_file, sheet_name)
        
        # Test different epsilon values
        eps_range = np.arange(0.1, 2.1, 0.1)
        evaluation_metrics = []
        
        for eps in eps_range:
            print(f"Testing DBSCAN with eps={eps:.1f}...")
            labels = perform_dbscan(X_weighted, eps, min_samples)
            metrics = evaluate_clustering(X_weighted, labels)
            evaluation_metrics.append(metrics)
        
        # Plot evaluation metrics
        plot_evaluation_metrics(eps_range, evaluation_metrics)
        
        # Perform final clustering with optimal parameters
        final_labels = perform_dbscan(X_weighted, optimal_eps, min_samples)
        metrics_df['DBSCAN'] = final_labels
        
        # Print final clustering results
        n_clusters, n_noise, silhouette_avg = evaluate_clustering(X_weighted, final_labels)
        print(f"\nFinal clustering results:")
        print(f"Number of clusters: {n_clusters}")
        print(f"Number of noise points: {n_noise}")
        print(f"Silhouette score: {silhouette_avg:.3f}")
        
        # Save results
        metrics_df.to_excel(input_file, index=False, sheet_name=sheet_name)
        print(f'\nClustering results saved to: {input_file}')
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()
