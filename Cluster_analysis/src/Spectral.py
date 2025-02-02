#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Spectral clustering for neuron calcium metrics data.
This script implements spectral clustering using RBF kernel for similarity matrix
and determines the optimal number of clusters using silhouette analysis.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import rbf_kernel

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
    'Decay Time': 1.5,
    'Rise Time': 1.5,
    'Latency': 1.0,
    'Frequency': 1.0
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

def compute_similarity_matrix(X: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Compute RBF kernel similarity matrix.
    
    Args:
        X: Input data matrix
        gamma: RBF kernel parameter
        
    Returns:
        Similarity matrix
    """
    return rbf_kernel(X, gamma=gamma)

def find_optimal_clusters(X: np.ndarray, similarity_matrix: np.ndarray, 
                        max_clusters: int = 10) -> tuple:
    """
    Find optimal number of clusters using silhouette analysis.
    
    Args:
        X: Input data matrix
        similarity_matrix: Pre-computed similarity matrix
        max_clusters: Maximum number of clusters to test
        
    Returns:
        Tuple of (optimal number of clusters, silhouette scores)
    """
    silhouette_scores = []
    range_n_clusters = range(2, max_clusters + 1)
    
    for n_clusters in range_n_clusters:
        print(f"Computing clustering for k={n_clusters}...")
        spectral = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            random_state=0
        )
        labels = spectral.fit_predict(similarity_matrix)
        silhouette_avg = silhouette_score(X, labels)
        silhouette_scores.append(silhouette_avg)
        print(f"Silhouette score: {silhouette_avg:.4f}")
    
    optimal_n_clusters = range_n_clusters[np.argmax(silhouette_scores)]
    return optimal_n_clusters, silhouette_scores

def plot_silhouette_scores(range_n_clusters: range, scores: list):
    """Plot silhouette scores for different numbers of clusters."""
    plt.figure(figsize=(8, 4))
    plt.plot(range_n_clusters, scores, 'bo-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Average Silhouette Score')
    plt.title('Silhouette Score vs Number of Clusters')
    plt.xticks(range_n_clusters)
    plt.show()

def main():
    """Main execution function."""
    # File paths and parameters
    input_file = '../datasets/Day3_Neuron_Calcium_Metrics.xlsx'
    sheet_name = 'Windows100_step10'
    gamma_value = 1.0
    
    try:
        # Load and preprocess data
        metrics_df, X_weighted = load_and_preprocess_data(input_file, sheet_name)
        
        # Compute similarity matrix
        similarity_matrix = compute_similarity_matrix(X_weighted, gamma=gamma_value)
        
        # Find optimal number of clusters
        optimal_n_clusters, silhouette_scores = find_optimal_clusters(
            X_weighted, 
            similarity_matrix
        )
        print(f"\nOptimal number of clusters: {optimal_n_clusters}")
        
        # Plot silhouette analysis results
        plot_silhouette_scores(range(2, 11), silhouette_scores)
        
        # Perform final clustering with optimal parameters
        spectral = SpectralClustering(
            n_clusters=optimal_n_clusters,
            affinity='precomputed',
            random_state=0
        )
        metrics_df['Spectral'] = spectral.fit_predict(similarity_matrix)
        
        # Save results
        metrics_df.to_excel(input_file, index=False, sheet_name=sheet_name)
        print(f'\nClustering results saved to: {input_file}')
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()
