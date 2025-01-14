import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, f_oneway
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
import os
import matplotlib

# Set Matplotlib style and configuration
sns.set_theme(style='darkgrid')  # Use seaborn's darkgrid theme
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'DejaVu Sans'

def create_directory(path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)

def cohens_d(x, y):
    """Calculate Cohen's d effect size"""
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std if pooled_std != 0 else np.nan

def eta_squared(f_stat, df_between, df_within):
    """Calculate Eta-squared effect size"""
    return f_stat * df_between / (f_stat * df_between + df_within)

def preprocess_data(input_file):
    """
    Preprocess data from the combined Excel file
    
    Parameters:
    - input_file: Path to Excel file containing neuronal activity, behavior, and timestamps
    
    Returns:
    - df: Original DataFrame
    - long_df: Long format DataFrame for analysis
    """
    # Read data
    print("Reading data file...")
    df = pd.read_excel(input_file)

    # Clean column names
    df.rename(columns=lambda x: x.strip(), inplace=True)

    # Get neuron columns (all columns containing 'n')
    neuron_cols = [col for col in df.columns if 'n' in col.lower()]
    if not neuron_cols:
        raise ValueError("No neuron columns found in the Excel file!")
    print(f"Found {len(neuron_cols)} neuron columns:", neuron_cols)

    # Create long format data
    print("Converting data to long format...")
    long_df = df.melt(id_vars=['stamp', 'behavior'], value_vars=neuron_cols,
                      var_name='NeuronID', value_name='Activity')
    print(f"Long format data rows: {long_df.shape[0]}")
    print("\nFirst few rows of long format data:")
    print(long_df.head())

    return df, long_df

def perform_statistical_analysis(long_df, output_dir):
    """
    Perform statistical analysis on neuronal activity across behavioral states
    
    Parameters:
    - long_df: Long format DataFrame containing neuronal activity and behavioral data
    - output_dir: Directory to save results
    """
    # Create output directory
    create_directory(output_dir)
    
    # Get unique neurons and behaviors
    neurons = long_df['NeuronID'].unique()
    behavior_states = long_df['behavior'].unique()
    num_states = len(behavior_states)
    print(f"\nFound {len(neurons)} neurons and {num_states} behavioral states")
    print("Behavioral states:", behavior_states)
    
    # Initialize results list
    results = []
    
    # Analyze each neuron
    for neuron in neurons:
        print(f"\nProcessing neuron: {neuron}")
        
        # Get neuron data
        neuron_data = long_df[long_df['NeuronID'] == neuron]
        groups = [neuron_data[neuron_data['behavior'] == state]['Activity'].dropna().values 
                 for state in behavior_states]
        
        # Skip if any group is empty
        if any(len(group) == 0 for group in groups):
            print(f"神经元 {neuron} 的某些行为状态下没有数据，跳过统计检验。")
            continue
        
        # Perform statistical test
        if num_states == 2:
            # Perform t-test for two groups
            group1, group2 = groups
            t_stat, p_val = ttest_ind(group1, group2, equal_var=False)  # Welch's t-test
            effect_size = cohens_d(group1, group2)
            test_type = 't-test'
            statistic = t_stat
        else:
            # Perform ANOVA for more than two groups
            try:
                f_stat, p_val = f_oneway(*groups)
                df_between = num_states - 1
                df_within = len(neuron_data) - num_states
                effect_size = eta_squared(f_stat, df_between, df_within)
                test_type = 'ANOVA'
                statistic = f_stat
            except Exception as e:
                print(f"神经元 {neuron} 的ANOVA检验出错: {e}")
                continue
        
        # Store results
        results.append({
            'NeuronID': neuron,
            'Test': test_type,
            'Statistic': statistic,
            'p-value': p_val,
            'EffectSize': effect_size
        })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Apply FDR correction
    if not results_df.empty:
        results_df.dropna(subset=['p-value'], inplace=True)
        results_df['p-adjusted'] = multipletests(results_df['p-value'], method='fdr_bh')[1]
        results_df['Significant'] = results_df['p-adjusted'] < 0.05
        
        # Save results
        results_df.to_excel(os.path.join(output_dir, 'statistical_analysis_results.xlsx'), index=False)
        significant_df = results_df[results_df['Significant']]
        significant_df.to_excel(os.path.join(output_dir, 'significant_neurons.xlsx'), index=False)
        
        # Plot results
        plot_statistical_results(results_df, significant_df, long_df, behavior_states, output_dir)
    
    return results_df

def plot_statistical_results(results_df, significant_df, long_df, behavior_states, output_dir):
    """
    Create visualizations for statistical analysis results
    """
    # 1. P-value distribution plot
    plt.figure(figsize=(10, 6))
    sns.histplot(data=results_df, x='p-adjusted', bins=30, kde=True)
    plt.axvline(0.05, color='red', linestyle='--', label='α = 0.05')
    plt.title('Distribution of Adjusted P-values')
    plt.xlabel('Adjusted P-value')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'p_value_distribution.png'), bbox_inches='tight')
    plt.close()
    
    # 2. Effect size distribution plot
    plt.figure(figsize=(10, 6))
    sns.histplot(data=results_df, x='EffectSize', bins=30)
    plt.title('Distribution of Effect Sizes')
    plt.xlabel('Effect Size')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, 'effect_size_distribution.png'), bbox_inches='tight')
    plt.close()
    
    # 3. Box plots for significant neurons
    for _, row in significant_df.iterrows():
        neuron = row['NeuronID']
        neuron_data = long_df[long_df['NeuronID'] == neuron]
        
        plt.figure(figsize=(10, 6))
        # Create boxplot
        sns.boxplot(x='behavior', y='Activity', data=neuron_data, color='lightgray')
        # Add stripplot instead of swarmplot
        sns.stripplot(x='behavior', y='Activity', data=neuron_data,
                     color='0.3', alpha=0.4, size=2, jitter=0.2)
        
        plt.title(f'Neuron {neuron} Activity Across Behavioral States\n' +
                 f'Adjusted p-value = {row["p-adjusted"]:.4e}, Effect Size = {row["EffectSize"]:.2f}')
        plt.xlabel('Behavioral State')
        plt.ylabel('Fluorescence Intensity')
        
        # Rotate x-axis labels if they are too long
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, f'{neuron}_behavior_comparison.png'), bbox_inches='tight')
        plt.close()
    
    # 4. Summary heatmap
    if not significant_df.empty:
        significant_neurons = significant_df['NeuronID'].tolist()
        activity_matrix = np.zeros((len(significant_neurons), len(behavior_states)))
        
        for i, neuron in enumerate(significant_neurons):
            for j, state in enumerate(behavior_states):
                activity_matrix[i, j] = np.mean(long_df[
                    (long_df['NeuronID'] == neuron) & 
                    (long_df['behavior'] == state)
                ]['Activity'])
        
        plt.figure(figsize=(12, len(significant_neurons) * 0.4 + 2))
        sns.heatmap(activity_matrix, xticklabels=behavior_states, yticklabels=significant_neurons,
                   cmap='coolwarm', center=0, annot=True, fmt='.2f')
        plt.title('Mean Activity of Significant Neurons Across Behavioral States')
        
        # Rotate x-axis labels if they are too long
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, 'significant_neurons_heatmap.png'), bbox_inches='tight')
        plt.close()

def main():
    # Set input/output paths
    input_file = '../../datasets/processed_Day3.xlsx'
    output_dir = '../../graph/statistical_analysis_day3'
    
    try:
        # Preprocess data
        print("\n1. Data preprocessing...")
        merged_df, long_df = preprocess_data(input_file)
        
        # Perform analysis
        print("\n2. Performing statistical analysis...")
        results = perform_statistical_analysis(long_df, output_dir)
        
        # Print summary
        if results is not None and not results.empty:
            significant_count = results['Significant'].sum()
            total_count = len(results)
            print(f"\nAnalysis Summary:")
            print(f"Total neurons analyzed: {total_count}")
            print(f"Significant neurons: {significant_count} ({significant_count/total_count*100:.1f}%)")
            print(f"Results saved in: {output_dir}")
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()