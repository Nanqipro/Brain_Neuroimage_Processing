import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =======================
# Configuration
# =======================

# File paths
DATA_FILE = '../../datasets/Day3_with_behavior_labels_filled.xlsx'
GRAPH_OUTPUT_DIR = '../../graph/mean_comparative_analysis_day3'

# Create output directory if it doesn't exist
os.makedirs(GRAPH_OUTPUT_DIR, exist_ok=True)

# Plot settings
plt.style.use('default')
plt.rcParams.update({
    'figure.figsize': (15, 8),
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.axisbelow': True,
    'axes.spines.top': False,
    'axes.spines.right': False
})

def load_data():
    """Load and validate the Excel data file."""
    try:
        df = pd.read_excel(DATA_FILE)
        print(f"Successfully loaded file: {DATA_FILE}")
        return df
    except FileNotFoundError:
        print(f"Error: File '{DATA_FILE}' not found.")
        exit(1)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        exit(1)

def get_neuron_columns(df):
    """Get available neuron columns from the dataframe."""
    neuron_columns = [f'n{i}' for i in range(1, 63)]
    available_columns = [col for col in neuron_columns if col in df.columns]
    
    if not available_columns:
        print("Error: No neuron columns found.")
        exit(1)
    
    print(f"Number of neuron columns: {len(available_columns)}")
    return available_columns

def plot_single_behavior(df, behavior, neuron_columns):
    """Plot and save analysis for a single behavior type."""
    # Create a copy to avoid SettingWithCopyWarning
    df_behavior = df[df['behavior'] == behavior].copy()
    
    if df_behavior.empty:
        print(f"Warning: No data found for behavior '{behavior}'")
        return
    
    # Handle missing values
    df_behavior.loc[:, neuron_columns] = df_behavior[neuron_columns].fillna(
        df_behavior[neuron_columns].mean()
    )
    
    # Calculate means
    neuron_means = df_behavior[neuron_columns].mean()
    
    # Create plot
    plt.figure(figsize=(15, 8))
    sns.barplot(x=neuron_means.index, y=neuron_means.values, color='skyblue')
    
    plt.title(f'Average Neuron Activity - {behavior}')
    plt.xlabel('Neuron')
    plt.ylabel('Average Activity')
    plt.xticks(rotation=90)
    
    # Save plot
    output_path = os.path.join(GRAPH_OUTPUT_DIR, f'single_behavior_{behavior.lower()}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_path}")

def plot_behavior_comparison(df, behavior1, behavior2, neuron_columns):
    """Plot and save comparison between two behaviors."""
    df_b1 = df[df['behavior'] == behavior1].copy()
    df_b2 = df[df['behavior'] == behavior2].copy()
    
    if df_b1.empty or df_b2.empty:
        print(f"Warning: Insufficient data for comparison {behavior1} vs {behavior2}")
        return
    
    # Calculate means
    means_b1 = df_b1[neuron_columns].mean()
    means_b2 = df_b2[neuron_columns].mean()
    
    # Create plot
    plt.figure(figsize=(15, 8))
    plt.bar(means_b1.index, means_b1.values, alpha=0.6, label=behavior1, color='blue')
    plt.bar(means_b2.index, means_b2.values, alpha=0.6, label=behavior2, color='red', width=0.4)
    
    plt.title(f'Neuron Activity Comparison: {behavior1} vs {behavior2}')
    plt.xlabel('Neuron')
    plt.ylabel('Average Activity')
    plt.xticks(rotation=90)
    plt.legend()
    
    # Save plot
    output_path = os.path.join(GRAPH_OUTPUT_DIR, 
                              f'comparison_{behavior1.lower()}_{behavior2.lower()}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_path}")

def plot_three_way_comparison(df, behaviors, neuron_columns):
    """Plot and save three-way behavior comparison."""
    if len(behaviors) < 2:
        print("Warning: Not enough behaviors for comparison")
        return
    
    # Generate all possible comparison pairs
    comparison_pairs = []
    for i, b1 in enumerate(behaviors[:-1]):
        for b2 in behaviors[i+1:]:
            comparison_pairs.append((b1, b2))
    
    # Only take the first 3 comparisons if there are more
    comparison_pairs = comparison_pairs[:3]
    num_comparisons = len(comparison_pairs)
    
    # Create subplot with the exact number of comparisons
    fig, axes = plt.subplots(1, num_comparisons, figsize=(6.5*num_comparisons, 6), sharey=True)
    
    # Ensure axes is always a list for consistent indexing
    if num_comparisons == 1:
        axes = [axes]
    
    for idx, (b1, b2) in enumerate(comparison_pairs):
        means_b1 = df[df['behavior'] == b1][neuron_columns].mean()
        means_b2 = df[df['behavior'] == b2][neuron_columns].mean()
        
        ax = axes[idx]
        ax.bar(means_b1.index, means_b1.values, color='blue', alpha=0.6, label=b1)
        ax.bar(means_b2.index, means_b2.values, color='red', alpha=0.6, label=b2, width=0.4)
        
        ax.set_title(f'{b1} vs {b2}')
        ax.tick_params(axis='x', rotation=90)
        ax.set_xlabel('Neuron')
        if idx == 0:  # Only set ylabel for the first subplot
            ax.set_ylabel('Average Activity')
        ax.legend()
    
    plt.tight_layout()
    output_path = os.path.join(GRAPH_OUTPUT_DIR, 'behavior_comparisons.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_path}")

def plot_exp_segments(df, neuron_columns):
    """Plot and save analysis for continuous Exp segments."""
    # Create segment groups
    df = df.copy()  # Create a copy to avoid SettingWithCopyWarning
    df['Exp_Group'] = (df['behavior'] != 'Exp').cumsum()
    df_exp = df[df['behavior'] == 'Exp'].copy()
    
    if df_exp.empty:
        print("Warning: No Exp segments found in data")
        return
    
    # Process each Exp segment
    for group_id, group_data in df_exp.groupby('Exp_Group'):
        group_means = group_data[neuron_columns].mean()
        start_index = group_data.index.min()
        end_index = group_data.index.max()
        
        plt.figure(figsize=(15, 8))
        plt.bar(group_means.index, group_means.values, color='skyblue')
        
        plt.title(f'Exp Segment {group_id} Analysis\n(Index Range: {start_index} - {end_index})')
        plt.xlabel('Neuron')
        plt.ylabel('Average Activity')
        plt.xticks(rotation=90)
        
        output_path = os.path.join(GRAPH_OUTPUT_DIR, f'exp_segment_{group_id}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {output_path}")

def main():
    """Main execution function."""
    # Load and prepare data
    df = load_data()
    neuron_columns = get_neuron_columns(df)
    
    # Get unique behaviors from data
    behaviors = sorted(df['behavior'].unique())
    print(f"Found behaviors in data: {behaviors}")
    
    # Generate all plots
    for behavior in behaviors:
        plot_single_behavior(df, behavior, neuron_columns)
    
    for i, b1 in enumerate(behaviors):
        for b2 in behaviors[i+1:]:
            plot_behavior_comparison(df, b1, b2, neuron_columns)
    
    if len(behaviors) >= 2:
        plot_three_way_comparison(df, behaviors, neuron_columns)
    plot_exp_segments(df, neuron_columns)

if __name__ == "__main__":
    main()
