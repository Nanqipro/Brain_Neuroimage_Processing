import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Load the Excel file
# file_path = './data/smoothed_trace_homecage.xlsx'
# file_path = './data/smoothed_normalized_2979_CSDS_Day6.xlsx'
file_path = '../../datasets/processed_Day6.xlsx'
df = pd.read_excel(file_path)

# Find behavior change points if 'behavior' column exists
if 'behavior' in df.columns:
    # Get indices where behavior changes
    behavior_changes = df.index[df['behavior'] != df['behavior'].shift()].tolist()
    # Get corresponding stamp values and behaviors
    change_stamps = df.loc[behavior_changes, 'stamp']
    behaviors = df.loc[behavior_changes, 'behavior']

# Set up the plot
plt.figure(figsize=(50, 15))

# Scaling factor to make the amplitude more visible
scaling_factor = 40

# Plot each line at a different vertical level, with scaling applied
for i in range(1, 64):
    column_name = f'n{i}'
    if column_name in df.columns:  # Only plot if the column exists
        plt.plot(df['stamp'], df[column_name] * scaling_factor + i * 50, label=column_name)

# Set x-axis limits to 0 to 500
plt.xlim(0,3000)

# Add vertical lines at behavior changes if behavior column exists
if 'behavior' in df.columns:
    for stamp, behavior in zip(change_stamps, behaviors):
        plt.axvline(x=stamp, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
        # Add behavior label
        plt.text(stamp, plt.ylim()[1], str(behavior), rotation=90, verticalalignment='top')

# Adding labels and title
plt.xlabel('Stamp')
plt.ylabel('Traces (n1 ~ n63)')
plt.title('Traces with Increased Amplitude')

# Adjust layout and show the plot
plt.tight_layout()
plt.savefig('../../graph/smooth_traces_amplitude_day6.png')  # Save the figure
# plt.show()
