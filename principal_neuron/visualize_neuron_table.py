import pandas as pd
import matplotlib.pyplot as plt
import io
import numpy as np

# --- Matplotlib Chinese Font Configuration ---
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # Default font: SimHei (Black Body)
    plt.rcParams['axes.unicode_minus'] = False  # Solve the issue of minus sign '-' displaying as a square
except Exception as e:
    print(f"字体配置失败，可能需要手动安装或指定可用中文字体: {e}")
    print("您可以尝试将 'SimHei' 替换为您系统中已有的中文字体，例如 'Microsoft YaHei', 'KaiTi' 等。")

# --- Data Loading and Preparation (from EMtrace_3标签版.csv) ---
csv_data_emtrace = """Behavior,Rank_1,Rank_2,Rank_3,Rank_4,Rank_5,Rank_6,Rank_7,Rank_8,Rank_9,Rank_10,Rank_11,Rank_12,Rank_13,Rank_14,Rank_15,Rank_16,Rank_17,Rank_18,Rank_19,Rank_20,Rank_21,Rank_22,Rank_23,Rank_24,Rank_25,Rank_26,Rank_27,Rank_28,Rank_29,Rank_30,Rank_31,Rank_32,Rank_33,Rank_34,Rank_35,Rank_36,Rank_37,Rank_38,Rank_39,Rank_40,Rank_41,Rank_42,Rank_43
Close,3,25,42,32,39,41,9,31,19,11,20,34,7,13,18,1,26,16,17,5,2,10,12,36,14,33,35,28,23,40,27,6,4,24,43,37,30,29,38,8,15,22,21
Middle,3,42,25,18,13,19,20,17,43,7,40,41,31,1,9,32,4,6,33,30,35,14,5,38,34,22,11,29,2,10,37,28,12,26,15,21,24,23,36,16,39,27,8
Open,5,39,30,10,22,18,27,43,16,40,32,17,38,11,4,29,13,26,21,19,6,15,36,34,25,2,20,33,37,42,23,9,8,12,14,24,35,31,41,28,1,7,3
"""
df_emtrace = pd.read_csv(io.StringIO(csv_data_emtrace))

top_n = 10
core_neurons_emtrace = {}
for index, row in df_emtrace.iterrows():
    behavior_name = row['Behavior']
    num_rank_cols = min(top_n, len(df_emtrace.columns) - 1)
    ranked_neurons = row[1:num_rank_cols+1].astype(int).tolist()
    core_neurons_emtrace[behavior_name] = ranked_neurons

print("--- EMtrace Top 10 Core Neurons ---") # English output
for behavior, neurons in core_neurons_emtrace.items():
    print(f"{behavior}: {neurons}")

# --- Calculate Neuron Sets for Categories ---
C_neurons = set(core_neurons_emtrace.get('Close', []))
M_neurons = set(core_neurons_emtrace.get('Middle', []))
O_neurons = set(core_neurons_emtrace.get('Open', []))

# Categories (English Names)
specific_C = C_neurons - (M_neurons | O_neurons)
specific_M = M_neurons - (C_neurons | O_neurons)
specific_O = O_neurons - (C_neurons | M_neurons)
shared_CM_only = (C_neurons & M_neurons) - O_neurons
shared_CO_only = (C_neurons & O_neurons) - M_neurons
shared_MO_only = (M_neurons & O_neurons) - C_neurons
shared_CMO = C_neurons & M_neurons & O_neurons

categories_dict = {
    "Specific to 'Close'": specific_C,
    "Specific to 'Middle'": specific_M,
    "Specific to 'Open'": specific_O,
    "Shared: 'C' & 'M' only": shared_CM_only,
    "Shared: 'C' & 'O' only": shared_CO_only,
    "Shared: 'M' & 'O' only": shared_MO_only,
    "Shared: 'C', 'M' & 'O'": shared_CMO,
}

print("\n--- Neuron Categories for Plot ---")
for name, neuron_set in categories_dict.items():
    ids_str = ', '.join(map(str, sorted(list(neuron_set)))) if neuron_set else "(None)"
    count = len(neuron_set)
    print(f"{name} (Count: {count}): {ids_str}")

# --- Generate Categorized Neuron ID List Plot ---
num_categories = len(categories_dict)
# Define layout: aim for 2 columns, adjust rows dynamically or set fixed
n_cols = 2
n_rows = (num_categories + n_cols - 1) // n_cols  # Calculate rows needed for 2 columns

fig_height = max(5, n_rows * 2) # Each category box gets ~2 inches height
fig_width = 12

fig, axs = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
fig.suptitle(f'Categorization of Top {top_n} Neurons (EMtrace Data)', fontsize=16, y=0.98)

axs_flat = axs.flatten()
category_items = list(categories_dict.items())

for i in range(num_categories):
    ax = axs_flat[i]
    ax.axis('off') # Turn off axis lines and ticks

    cat_name, neuron_set = category_items[i]
    ids_list = sorted(list(neuron_set))
    count = len(ids_list)
    
    title_text = f"{cat_name}\n(Count: {count})"
    
    # Neuron IDs string, with wrapping for display
    ids_per_line = 6 # Number of IDs before trying to wrap
    display_ids_str = ""
    if not ids_list:
        display_ids_str = "(None)"
    else:
        for k in range(0, len(ids_list), ids_per_line):
            display_ids_str += ', '.join(map(str, ids_list[k:k+ids_per_line])) + "\n"
        display_ids_str = display_ids_str.strip() # Remove trailing newline

    # Display text in the center of the subplot/cell
    ax.text(0.5, 0.7, title_text, ha='center', va='top', fontsize=12, weight='bold', wrap=True)
    ax.text(0.5, 0.45, display_ids_str, ha='center', va='top', fontsize=10, wrap=True, linespacing=1.5)

    # Add a border around the subplot for visual separation
    rect = plt.Rectangle((0,0), 1, 1, transform=ax.transAxes, fill=False, edgecolor='#cccccc', linewidth=1)
    ax.add_patch(rect)

# If there's an odd number of categories, the last subplot in the grid might be unused.
# Hide it if it exists and is beyond our number of categories.
if num_categories < len(axs_flat):
    for j in range(num_categories, len(axs_flat)):
        axs_flat[j].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to make space for suptitle

output_filename = "neuron_categorization_plot_emtrace.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"\nCategorization plot saved as: {output_filename}")
plt.show()

print("\nScript finished.") 