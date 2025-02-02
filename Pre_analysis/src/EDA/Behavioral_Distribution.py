import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """Load Day6 dataset"""
    return pd.read_excel('../../datasets/Day3_with_behavior_labels_filled.xlsx')

def analyze_behavior_distribution(data):
    """Analyze and visualize behavior distribution for Day3 dataset"""
    # Filter out CD1 behavior
    data_filtered = data[data['behavior'] != 'CD1']
    
    # Count the frequency of each behavior
    behavior_counts = data_filtered['behavior'].value_counts()
    
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar plot
    sns.barplot(x=behavior_counts.index, y=behavior_counts.values, ax=ax1)
    ax1.set_title('Behavior Distribution (Day 3, excluding CD1) - Bar Plot')
    ax1.set_xlabel('Behavior Type')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=45)
    
    # Pie chart
    ax2.pie(behavior_counts.values, labels=behavior_counts.index, autopct='%1.1f%%')
    ax2.set_title('Behavior Distribution (Day 3, excluding CD1) - Pie Chart')
    
    plt.tight_layout()
    plt.savefig('../../graph/behavior_distribution_day3_no_cd1.png')
    plt.close()
    
    # Print numerical summary
    print("\nBehavior Distribution for Day 3 (excluding CD1):")
    print(behavior_counts)
    print("\nPercentage Distribution:")
    print((behavior_counts / len(data_filtered) * 100).round(2), "%")

def main():
    # Load data
    day6_data = load_data()
    
    # Analyze Day6
    analyze_behavior_distribution(day6_data)

if __name__ == "__main__":
    main()
