#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PCA analysis for neuron calcium metrics data.
This script performs dimensionality reduction using PCA and visualizes 
the results in 2D space with cluster coloring.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

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

CLUSTER_COLUMN = 'k-means-Manhattan'  # 可替换为其他聚类结果列名
PCA_COMPONENTS = 2

def load_data(file_path: str, sheet_name: str) -> pd.DataFrame:
    """
    Load data from Excel file.
    
    Args:
        file_path: Path to the Excel file
        sheet_name: Name of the sheet to read
        
    Returns:
        Loaded DataFrame
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    return df.dropna(subset=FEATURES).reset_index(drop=True)

def perform_pca(df: pd.DataFrame) -> tuple:
    """
    Perform PCA dimensionality reduction.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of (DataFrame with PCA results, PCA model)
    """
    X = df[FEATURES].values
    pca = PCA(n_components=PCA_COMPONENTS)
    X_pca = pca.fit_transform(X)
    
    df_pca = df.copy()
    df_pca['PCA-1'] = X_pca[:, 0]
    df_pca['PCA-2'] = X_pca[:, 1]
    
    return df_pca, pca

def plot_pca_results(df: pd.DataFrame, pca: PCA):
    """
    Visualize PCA results with cluster coloring.
    
    Args:
        df: DataFrame containing PCA coordinates and cluster labels
        pca: PCA model object
    """
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='PCA-1', y='PCA-2',
        data=df,
        hue=CLUSTER_COLUMN,
        palette='viridis',
        legend='full'
    )
    plt.title("PCA Projection of Neuron Calcium Metrics")
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%})")
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%})")
    plt.show()

def save_results(df: pd.DataFrame, file_path: str, sheet_name: str):
    """
    Save results back to Excel file.
    
    Args:
        df: DataFrame to save
        file_path: Output file path
        sheet_name: Sheet name to use
    """
    df.to_excel(file_path, index=False, sheet_name=sheet_name)

def main():
    """Main execution function."""
    # Configuration
    input_file = '../datasets/Day3_Neuron_Calcium_Metrics.xlsx'
    sheet_name = 'Windows100_step10'
    
    try:
        # Load and preprocess data
        metrics_df = load_data(input_file, sheet_name)
        
        # Perform PCA
        metrics_pca, pca = perform_pca(metrics_df)
        
        # Visualize results
        plot_pca_results(metrics_pca, pca)
        
        # Save results
        save_results(metrics_pca, input_file, sheet_name)
        print(f'PCA results saved to: {input_file}')
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()
