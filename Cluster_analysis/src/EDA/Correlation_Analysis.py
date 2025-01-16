import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

def load_data(file_path):
    """Load data from Excel file"""
    data = pd.read_excel(file_path)
    # Assuming behavior labels are in the last column
    neuron_data = data.iloc[:, :-1]  # All columns except the last one
    behavior_labels = data.iloc[:, -1]  # Last column
    return neuron_data, behavior_labels

def compute_neuron_correlations(neuron_data):
    """Compute correlations between neurons using Pearson correlation"""
    corr_matrix = neuron_data.corr(method='pearson')
    return corr_matrix

def plot_correlation_heatmap(corr_matrix, title, save_path):
    """Plot correlation matrix as a heatmap"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, vmin=-1, vmax=1)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def compute_neuron_behavior_correlation(neuron_data, behavior_labels):
    """Compute correlation between neurons and behavior"""
    n_neurons = neuron_data.shape[1]
    correlations = []
    p_values = []
    
    # If behavior is categorical, use point-biserial correlation
    if behavior_labels.dtype == 'object' or behavior_labels.nunique() < 5:
        # Convert behavior to numeric if categorical
        behavior_numeric = pd.Categorical(behavior_labels).codes
        
        for i in range(n_neurons):
            corr, p_val = stats.pointbiserialr(neuron_data.iloc[:, i], behavior_numeric)
            correlations.append(corr)
            p_values.append(p_val)
    else:
        # Use Pearson correlation for continuous behavior
        for i in range(n_neurons):
            corr, p_val = stats.pearsonr(neuron_data.iloc[:, i], behavior_labels)
            correlations.append(corr)
            p_values.append(p_val)
    
    return pd.DataFrame({
        'Neuron': [f'Neuron_{i+1}' for i in range(n_neurons)],
        'Correlation': correlations,
        'P_value': p_values
    })

def plot_neuron_behavior_heatmap(behavior_corr, save_path):
    """Plot neuron-behavior correlation as a heatmap"""
    # Create a DataFrame with just the correlations
    corr_data = behavior_corr[['Neuron', 'Correlation']].copy()
    # Reshape data for heatmap (1 row matrix)
    corr_matrix = pd.DataFrame([corr_data['Correlation'].values], 
                             columns=corr_data['Neuron'])
    
    plt.figure(figsize=(15, 3))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, vmin=-1, vmax=1,
                cbar_kws={'label': 'Correlation Coefficient'},
                xticklabels=True, yticklabels=['Behavior'])
    plt.title('Neuron-Behavior Correlation Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # File paths
    input_file = '../../datasets/Day9_with_behavior_labels_filled.xlsx'
    output_dir = '../../graph/'
    
    # Load data
    print("Loading data...")
    neuron_data, behavior_labels = load_data(input_file)
    
    # Compute and plot neuron-to-neuron correlations
    print("Computing neuron-to-neuron correlations...")
    neuron_corr = compute_neuron_correlations(neuron_data)
    plot_correlation_heatmap(
        neuron_corr, 
        'Neuron-to-Neuron Correlation Matrix', 
        f'{output_dir}neuron_correlation_heatmap_day9.png'
    )
    
    # Compute neuron-to-behavior correlations
    print("Computing neuron-to-behavior correlations...")
    behavior_corr = compute_neuron_behavior_correlation(neuron_data, behavior_labels)
    
    # Sort neurons by absolute correlation value
    behavior_corr['Abs_Correlation'] = abs(behavior_corr['Correlation'])
    behavior_corr_sorted = behavior_corr.sort_values('Abs_Correlation', ascending=False)
    
    # Print top correlated neurons
    print("\nTop 20 neurons most correlated with behavior:")
    print(behavior_corr_sorted.head(20))
    
    # Save results to Excel
    behavior_corr_sorted.to_excel(f'../../datasets/neuron_behavior_correlations_day9.xlsx', index=False)
    
    # Plot neuron-behavior correlations as bar plot
    plt.figure(figsize=(15, 6))
    x = range(len(behavior_corr_sorted))
    plt.bar(x, behavior_corr_sorted['Correlation'])
    
    # Add neuron labels
    plt.xticks(x, behavior_corr_sorted['Neuron'], rotation=45, ha='right')
    plt.xlabel('Neurons')
    plt.ylabel('Correlation with Behavior')
    plt.title('Neuron-Behavior Correlations')
    plt.tight_layout()
    plt.savefig(f'{output_dir}neuron_behavior_correlation_day9.png')
    plt.close()
    
    # Plot neuron-behavior correlations as heatmap
    plot_neuron_behavior_heatmap(
        behavior_corr_sorted,
        f'{output_dir}neuron_behavior_heatmap_day9.png'
    )

if __name__ == "__main__":
    main()
