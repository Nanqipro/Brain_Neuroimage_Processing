#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
K-means clustering analysis for neuron calcium metrics data.
This script performs weighted k-means clustering on neuron activity data
and determines the optimal number of clusters using elbow and silhouette methods.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
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
        Preprocessed DataFrame
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    if df[FEATURES].isnull().values.any():
        print("Warning: Data contains missing values, removing rows with NaN...")
        df = df.dropna(subset=FEATURES).reset_index(drop=True)
    
    return df

def scale_and_weight_features(df: pd.DataFrame) -> np.ndarray:
    """
    Standardize features and apply weights.
    
    Args:
        df: Input DataFrame with feature columns
        
    Returns:
        Weighted and scaled feature matrix
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURES])
    
    weight_values = np.array([FEATURE_WEIGHTS[feature] for feature in FEATURES])
    return X_scaled * weight_values

def plot_elbow_method(X_weighted: np.ndarray):
    """Plot elbow method for determining optimal k."""
    distortions = []
    K = range(1, 11)
    
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(X_weighted)
        distortions.append(kmeans.inertia_)
    
    plt.figure(figsize=(8, 4))
    plt.plot(K, distortions, 'bo-')
    plt.xlabel('Cluster Number (k)')
    plt.ylabel('SSE (Sum of Squared Errors)')
    plt.title('Elbow Method')
    plt.xticks(K)
    plt.show()

def plot_silhouette_method(X_weighted: np.ndarray):
    """Plot silhouette method for determining optimal k."""
    silhouette_scores = []
    K = range(2, 11)
    
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(X_weighted)
        score = silhouette_score(X_weighted, labels)
        silhouette_scores.append(score)
    
    plt.figure(figsize=(8, 4))
    plt.plot(K, silhouette_scores, 'bo-')
    plt.xlabel('Cluster Number (k)')
    plt.ylabel('Silhouette Coefficient')
    plt.title('Silhouette Coefficient Method')
    plt.xticks(K)
    plt.show()

def main():
    """Main execution function."""
    # File paths and parameters
    input_file = '../datasets/Day3_Neuron_Calcium_Metrics.xlsx'
    sheet_name = 'Windows100_step10'
    optimal_k = 6  # Optimal number of clusters
    
    try:
        # Load and preprocess data
        metrics_df = load_and_preprocess_data(input_file, sheet_name)
        
        # Scale and weight features
        X_weighted = scale_and_weight_features(metrics_df)
        
        # Plot methods for determining optimal k
        plot_elbow_method(X_weighted)
        plot_silhouette_method(X_weighted)
        
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=0)
        metrics_df['k-means-ed'] = kmeans.fit_predict(X_weighted)
        
        # Save results
        metrics_df.to_excel(input_file, index=False, sheet_name=sheet_name)
        print(f'Clustering results saved to: {input_file}')
        
    except FileNotFoundError:
        print(f"Error: Could not find file {input_file}")
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()
