#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gaussian Mixture Model (GMM) clustering for neuron calcium metrics data.
This script implements GMM clustering algorithm and determines the optimal 
number of components using various metrics.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy.stats import normaltest

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

def perform_gmm(X: np.ndarray, n_components: int, random_state: int = 0) -> tuple:
    """
    Perform GMM clustering.
    
    Args:
        X: Input data matrix
        n_components: Number of Gaussian components
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (labels, model)
    """
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    labels = gmm.fit_predict(X)
    return labels, gmm

def evaluate_clustering(X: np.ndarray, labels: np.ndarray, gmm: GaussianMixture) -> tuple:
    """
    Evaluate clustering results.
    
    Args:
        X: Input data matrix
        labels: Cluster labels
        gmm: Fitted GMM model
        
    Returns:
        Tuple of (BIC score, AIC score, silhouette score)
    """
    bic = gmm.bic(X)
    aic = gmm.aic(X)
    silhouette_avg = silhouette_score(X, labels)
    return bic, aic, silhouette_avg

def plot_evaluation_metrics(K: range, metrics: list):
    """Plot evaluation metrics for different numbers of components."""
    bic_scores, aic_scores, silhouette_scores = zip(*metrics)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    
    ax1.plot(K, bic_scores, 'bo-')
    ax1.set_xlabel('Number of Components')
    ax1.set_ylabel('BIC Score')
    ax1.set_title('BIC Score vs Components')
    
    ax2.plot(K, aic_scores, 'ro-')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('AIC Score')
    ax2.set_title('AIC Score vs Components')
    
    ax3.plot(K, silhouette_scores, 'go-')
    ax3.set_xlabel('Number of Components')
    ax3.set_ylabel('Silhouette Score')
    ax3.set_title('Silhouette Score vs Components')
    
    plt.tight_layout()
    plt.show()

def test_normality(X: np.ndarray):
    """Test for normality in each feature."""
    for i, feature in enumerate(FEATURES):
        statistic, p_value = normaltest(X[:, i])
        print(f"{feature}: p-value = {p_value:.4f}")

def main():
    """Main execution function."""
    # File paths and parameters
    input_file = '../datasets/Day3_Neuron_Calcium_Metrics.xlsx'
    sheet_name = 'Windows100_step10'
    optimal_components = 6
    
    try:
        # Load and preprocess data
        metrics_df, X_weighted = load_and_preprocess_data(input_file, sheet_name)
        
        # Test normality of features
        print("Testing normality of features:")
        test_normality(X_weighted)
        
        # Test different numbers of components
        evaluation_metrics = []
        K = range(2, 11)
        
        for k in K:
            print(f"\nTesting GMM with {k} components...")
            labels, gmm = perform_gmm(X_weighted, k)
            metrics = evaluate_clustering(X_weighted, labels, gmm)
            evaluation_metrics.append(metrics)
        
        # Plot evaluation metrics
        plot_evaluation_metrics(K, evaluation_metrics)
        
        # Perform final clustering with optimal parameters
        final_labels, _ = perform_gmm(X_weighted, optimal_components)
        metrics_df['GMM'] = final_labels
        
        # Save results
        metrics_df.to_excel(input_file, index=False, sheet_name=sheet_name)
        print(f'\nClustering results saved to: {input_file}')
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()
