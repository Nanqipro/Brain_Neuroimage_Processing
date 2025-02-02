import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mutual_info_score
from sklearn.cross_decomposition import PLSRegression, CCA
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import grangercausalitytests

class CorrelationAnalyzer:
    def __init__(self, data_path, output_dir):
        self.data_path = data_path
        self.output_dir = output_dir
        self.neuron_data = None
        self.behavior_labels = None
        self.load_data()

    def load_data(self):
        """Load data from Excel file"""
        data = pd.read_excel(self.data_path)
        self.neuron_data = data.iloc[:, :-1]  # All columns except the last one
        
        # Convert behavior labels to numeric
        behavior_column = data.iloc[:, -1]
        if behavior_column.dtype == 'object':
            # If categorical, convert to numeric codes but keep as pandas Series
            categories = pd.Categorical(behavior_column)
            self.behavior_labels = pd.Series(categories.codes, index=behavior_column.index)
            print("Converted categorical behavior labels to numeric codes")
            print("Behavior categories:", dict(enumerate(categories.categories)))
        else:
            # Try to convert to numeric, replacing non-numeric values with NaN
            self.behavior_labels = pd.to_numeric(behavior_column, errors='coerce')
            # Remove any rows where behavior is NaN
            valid_mask = ~self.behavior_labels.isna()
            self.neuron_data = self.neuron_data[valid_mask]
            self.behavior_labels = self.behavior_labels[valid_mask]
            print("Converted behavior labels to numeric values")
        
        print(f"Loaded data with {self.neuron_data.shape[1]} neurons and {len(self.behavior_labels)} samples")
        print(f"Behavior labels range: [{self.behavior_labels.min()}, {self.behavior_labels.max()}]")
        print(f"Unique behavior values: {sorted(self.behavior_labels.unique())}")
        print(f"Behavior value counts:\n{self.behavior_labels.value_counts().sort_index()}")

    def pearson_correlation(self):
        """Compute Pearson correlation"""
        correlations = []
        p_values = []
        for i in range(self.neuron_data.shape[1]):
            corr, p_val = stats.pearsonr(self.neuron_data.iloc[:, i], self.behavior_labels)
            correlations.append(corr)
            p_values.append(p_val)
        return pd.DataFrame({
            'Neuron': [f'Neuron_{i+1}' for i in range(len(correlations))],
            'Pearson_Correlation': correlations,
            'P_value': p_values
        })

    def spearman_correlation(self):
        """Compute Spearman rank correlation"""
        correlations = []
        p_values = []
        for i in range(self.neuron_data.shape[1]):
            corr, p_val = stats.spearmanr(self.neuron_data.iloc[:, i], self.behavior_labels)
            correlations.append(corr)
            p_values.append(p_val)
        return pd.DataFrame({
            'Neuron': [f'Neuron_{i+1}' for i in range(len(correlations))],
            'Spearman_Correlation': correlations,
            'P_value': p_values
        })

    def kendall_correlation(self):
        """Compute Kendall's Tau correlation"""
        correlations = []
        p_values = []
        for i in range(self.neuron_data.shape[1]):
            corr, p_val = stats.kendalltau(self.neuron_data.iloc[:, i], self.behavior_labels)
            correlations.append(corr)
            p_values.append(p_val)
        return pd.DataFrame({
            'Neuron': [f'Neuron_{i+1}' for i in range(len(correlations))],
            'Kendall_Correlation': correlations,
            'P_value': p_values
        })

    def mutual_information(self):
        """Compute Mutual Information"""
        mi_scores = []
        n_bins = 10  # Number of bins for discretization
        
        for i in range(self.neuron_data.shape[1]):
            # Discretize continuous neuron data using bins
            neuron_data_binned = pd.qcut(self.neuron_data.iloc[:, i], 
                                       q=n_bins, 
                                       labels=False, 
                                       duplicates='drop')
            
            # Compute mutual information using discretized data
            mi = mutual_info_score(neuron_data_binned, self.behavior_labels)
            mi_scores.append(mi)
            
        # Normalize MI scores to [0, 1] range
        mi_scores = np.array(mi_scores)
        if mi_scores.max() > 0:
            mi_scores = mi_scores / mi_scores.max()
            
        return pd.DataFrame({
            'Neuron': [f'Neuron_{i+1}' for i in range(len(mi_scores))],
            'Mutual_Information': mi_scores
        })

    def plot_correlation_heatmap(self, correlation_data, method_name):
        """Plot correlation heatmap for a specific method"""
        plt.figure(figsize=(15, 3))
        corr_matrix = pd.DataFrame([correlation_data.iloc[:, 1].values], 
                                 columns=correlation_data['Neuron'])
        
        # Adjust vmin and vmax based on the correlation method
        if method_name == 'Mutual_Information':
            vmin, vmax = 0, 1  # MI is normalized to [0, 1]
            center = None
        else:
            vmin, vmax = -1, 1  # Other correlations are in [-1, 1]
            center = 0
            
        sns.heatmap(corr_matrix, cmap='coolwarm', center=center, vmin=vmin, vmax=vmax,
                   cbar_kws={'label': f'{method_name} Coefficient'},
                   xticklabels=True, yticklabels=['Behavior'])
        plt.title(f'Neuron-Behavior {method_name} Matrix')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'../../graph/neuron_behavior_{method_name.lower()}_heatmap_day6.png')
        plt.close()

    def plot_correlation_barplot(self, correlation_data, method_name):
        """Plot correlation bar plot for a specific method"""
        plt.figure(figsize=(15, 6))
        x = range(len(correlation_data))
        plt.bar(x, correlation_data.iloc[:, 1])
        
        # Add neuron labels
        plt.xticks(x, correlation_data['Neuron'], rotation=45, ha='right')
        plt.xlabel('Neurons')
        plt.ylabel(f'{method_name} with Behavior')
        plt.title(f'Neuron-Behavior {method_name}')
        plt.tight_layout()
        plt.savefig(f'../../graph/neuron_behavior_{method_name.lower()}_barplot_day6.png')
        plt.close()

    def analyze_all_correlations(self):
        """Perform all correlation analyses and combine results"""
        print("Computing correlations...")
        
        # Compute all correlations
        pearson_results = self.pearson_correlation()
        spearman_results = self.spearman_correlation()
        kendall_results = self.kendall_correlation()
        mi_results = self.mutual_information()
        
        # Merge all results
        results = pearson_results.merge(spearman_results[['Neuron', 'Spearman_Correlation']], on='Neuron')
        results = results.merge(kendall_results[['Neuron', 'Kendall_Correlation']], on='Neuron')
        results = results.merge(mi_results, on='Neuron')
        
        # Add absolute correlation values for sorting
        results['Abs_Pearson'] = abs(results['Pearson_Correlation'])
        results = results.sort_values('Abs_Pearson', ascending=False)
        
        # Save results
        results.to_excel('../../datasets/comprehensive_correlations_day6.xlsx', index=False)
        
        # Plot heatmaps and bar plots for each correlation method
        # Pearson
        pearson_data = results[['Neuron', 'Pearson_Correlation']].sort_values('Pearson_Correlation', ascending=False)
        self.plot_correlation_heatmap(pearson_data, 'Pearson')
        self.plot_correlation_barplot(pearson_data, 'Pearson Correlation')
        
        # Spearman
        spearman_data = results[['Neuron', 'Spearman_Correlation']].sort_values('Spearman_Correlation', ascending=False)
        self.plot_correlation_heatmap(spearman_data, 'Spearman')
        self.plot_correlation_barplot(spearman_data, 'Spearman Correlation')
        
        # Kendall
        kendall_data = results[['Neuron', 'Kendall_Correlation']].sort_values('Kendall_Correlation', ascending=False)
        self.plot_correlation_heatmap(kendall_data, 'Kendall')
        self.plot_correlation_barplot(kendall_data, 'Kendall Correlation')
        
        # Mutual Information
        mi_data = results[['Neuron', 'Mutual_Information']].sort_values('Mutual_Information', ascending=False)
        self.plot_correlation_heatmap(mi_data, 'Mutual_Information')
        self.plot_correlation_barplot(mi_data, 'Mutual Information')
        
        # Print top correlated neurons
        print("\nTop 20 neurons most correlated with behavior (Pearson):")
        print(results[['Neuron', 'Pearson_Correlation', 'P_value']].head(20))
        
        return results

def main():
    # Initialize analyzer
    analyzer = CorrelationAnalyzer(
        data_path='../../datasets/processed_Day6.xlsx',
        output_dir='../../datasets/'  # This is kept for potential future use
    )
    
    # Perform comprehensive correlation analysis
    results = analyzer.analyze_all_correlations()

if __name__ == "__main__":
    main()
