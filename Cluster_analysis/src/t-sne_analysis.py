import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

def run_tsne_analysis(file_path):
    """
    Performs t-SNE analysis on neuron calcium metrics data.
    
    Args:
        file_path (str): Path to the Excel file containing the metrics data.
    """
    # Load data
    metrics_df = pd.read_excel(file_path, sheet_name='Windows100_step10')
    
    # Define features for t-SNE
    features = ['Start Time', 'Amplitude', 'Peak', 'Decay Time', 'Rise Time', 'Latency', 'Frequency'] 
    X = metrics_df[features].values

    # Perform t-SNE to reduce dimensionality to 2D
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=0)
    tsne_results = tsne.fit_transform(X)
    
    # Add t-SNE results to original dataframe
    metrics_df['t-SNE-1'] = tsne_results[:, 0]
    metrics_df['t-SNE-2'] = tsne_results[:, 1]
    
    # Visualize t-SNE results
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='t-SNE-1', y='t-SNE-2',
        hue='k-means-Hausdorff', 
        data=metrics_df,
        palette='viridis',
        legend='full'
    )
    plt.title("t-SNE Clustering of Neuron Calcium Metrics")
    plt.xlabel("t-SNE Dimension 1") 
    plt.ylabel("t-SNE Dimension 2")
    plt.show()

    # Save results to file  
    metrics_df.to_excel(file_path, index=False, sheet_name='Windows100_step10')
    print(f't-SNE clustering results saved to: {file_path}')

if __name__ == '__main__':
    data_file_path = '../datasets/Day3_Neuron_Calcium_Metrics.xlsx'
    run_tsne_analysis(data_file_path)
