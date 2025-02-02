import pandas as pd
from umap import UMAP
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple
import numpy as np


def load_data(file_path: str, sheet_name: str) -> pd.DataFrame:
    """
    Load neuron calcium metrics data from an Excel file.

    Args:
        file_path (str): Path to the Excel file containing the metrics data.
        sheet_name (str): Name of the sheet containing the data.

    Returns:
        pd.DataFrame: Loaded data as a pandas DataFrame.
    """
    return pd.read_excel(file_path, sheet_name=sheet_name)


def prepare_features(data: pd.DataFrame, feature_columns: List[str]) -> np.ndarray:
    """
    Extract feature columns from the DataFrame.

    Args:
        data (pd.DataFrame): Input DataFrame containing all metrics.
        feature_columns (List[str]): List of column names to use as features.

    Returns:
        np.ndarray: Array of feature values.
    """
    return data[feature_columns].values


def perform_umap(features: np.ndarray, n_components: int = 2, 
                n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
    """
    Perform UMAP dimensionality reduction.

    Args:
        features (np.ndarray): Input feature array.
        n_components (int, optional): Number of dimensions to reduce to. Defaults to 2.
        n_neighbors (int, optional): Number of neighbors to consider. Defaults to 15.
        min_dist (float, optional): Minimum distance parameter. Defaults to 0.1.

    Returns:
        np.ndarray: Reduced dimensional representation of the data.
    """
    umap_model = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42  # 固定随机种子以确保结果可重复
    )
    return umap_model.fit_transform(features)


def visualize_umap(data: pd.DataFrame, cluster_column: str = 'k-means-Hausdorff',
                  title: str = "UMAP Clustering of Neuron Calcium Metrics") -> None:
    """
    Create and display a scatter plot of UMAP results.

    Args:
        data (pd.DataFrame): DataFrame containing UMAP results and cluster labels.
        cluster_column (str, optional): Column name for cluster labels. Defaults to 'k-means-Hausdorff'.
        title (str, optional): Plot title. Defaults to "UMAP Clustering of Neuron Calcium Metrics".
    """
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='UMAP-1',
        y='UMAP-2',
        hue=cluster_column,
        data=data,
        palette='viridis',
        legend='full'
    )
    plt.title(title)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.show()


def save_results(data: pd.DataFrame, file_path: str, sheet_name: str) -> None:
    """
    Save the results back to an Excel file.

    Args:
        data (pd.DataFrame): DataFrame containing the results.
        file_path (str): Path to save the Excel file.
        sheet_name (str): Name of the sheet to save the data to.
    """
    data.to_excel(file_path, index=False, sheet_name=sheet_name)
    print(f'UMAP clustering results saved to: {file_path}')


def run_umap_analysis(file_path: str, sheet_name: str = 'Windows100_step10') -> None:
    """
    Run the complete UMAP analysis pipeline.

    Args:
        file_path (str): Path to the Excel file containing the metrics data.
        sheet_name (str, optional): Name of the sheet containing the data. 
                                  Defaults to 'Windows100_step10'.
    """
    # 定义要使用的特征列
    feature_columns = [
        'Start Time', 'Amplitude', 'Peak', 
        'Decay Time', 'Rise Time', 'Latency', 'Frequency'
    ]

    # 加载数据
    metrics_df = load_data(file_path, sheet_name)

    # 准备特征数据
    features = prepare_features(metrics_df, feature_columns)

    # 执行UMAP降维
    umap_results = perform_umap(features)

    # 将UMAP结果添加到原始数据框
    metrics_df['UMAP-1'] = umap_results[:, 0]
    metrics_df['UMAP-2'] = umap_results[:, 1]

    # 可视化结果
    visualize_umap(metrics_df)

    # 保存结果
    save_results(metrics_df, file_path, sheet_name)


if __name__ == '__main__':
    # 设置输入文件路径
    input_file_path = '../datasets/Day3_Neuron_Calcium_Metrics.xlsx'
    
    # 运行分析
    run_umap_analysis(input_file_path)