import pandas as pd
import matplotlib.pyplot as plt
import io
import numpy as np
import random
import matplotlib.patches as mpatches # For custom legend shapes

# --- Data Loading and Preparation (from EMtrace_3标签版.csv) ---
csv_data_emtrace = """Behavior,Rank_1,Rank_2,Rank_3,Rank_4,Rank_5,Rank_6,Rank_7,Rank_8,Rank_9,Rank_10,Rank_11,Rank_12,Rank_13,Rank_14,Rank_15,Rank_16,Rank_17,Rank_18,Rank_19,Rank_20,Rank_21,Rank_22,Rank_23,Rank_24,Rank_25,Rank_26,Rank_27,Rank_28,Rank_29,Rank_30,Rank_31,Rank_32,Rank_33,Rank_34,Rank_35,Rank_36,Rank_37,Rank_38,Rank_39,Rank_40,Rank_41,Rank_42,Rank_43
Close,3,25,42,32,39,41,9,31,19,11,20,34,7,13,18,1,26,16,17,5,2,10,12,36,14,33,35,28,23,40,27,6,4,24,43,37,30,29,38,8,15,22,21
Middle,3,42,25,18,13,19,20,17,43,7,40,41,31,1,9,32,4,6,33,30,35,14,5,38,34,22,11,29,2,10,37,28,12,26,15,21,24,23,36,16,39,27,8
Open,5,39,30,10,22,18,27,43,16,40,32,17,38,11,4,29,13,26,21,19,6,15,36,34,25,2,20,33,37,42,23,9,8,12,14,24,35,31,41,28,1,7,3
"""
df_emtrace = pd.read_csv(io.StringIO(csv_data_emtrace))

top_n = 10 # Changed to Top 10 as per previous request
core_neurons_emtrace = {}
for index, row in df_emtrace.iterrows():
    behavior_name = row['Behavior']
    num_rank_cols = min(top_n, len(df_emtrace.columns) - 1)
    ranked_neurons = row[1:num_rank_cols+1].astype(int).tolist()
    core_neurons_emtrace[behavior_name] = ranked_neurons

print(f"--- EMtrace Top {top_n} Core Neurons ---")
for behavior, neurons in core_neurons_emtrace.items():
    print(f"{behavior}: {neurons}")

C_neurons = set(core_neurons_emtrace.get('Close', []))
M_neurons = set(core_neurons_emtrace.get('Middle', []))
O_neurons = set(core_neurons_emtrace.get('Open', []))
all_unique_neurons = sorted(list(C_neurons | M_neurons | O_neurons))

color_map_rgb = {
    'Close': np.array([1, 0, 0]),  # Red
    'Middle': np.array([0, 1, 0]), # Green
    'Open': np.array([0, 0, 1])    # Blue
}

neuron_attributes = []
for nid in all_unique_neurons:
    is_C = nid in C_neurons
    is_M = nid in M_neurons
    is_O = nid in O_neurons
    final_color_vec = np.array([0,0,0], dtype=float)
    behaviors_involved_abbr = []
    if is_C: final_color_vec += color_map_rgb['Close']; behaviors_involved_abbr.append('C')
    if is_M: final_color_vec += color_map_rgb['Middle']; behaviors_involved_abbr.append('M')
    if is_O: final_color_vec += color_map_rgb['Open']; behaviors_involved_abbr.append('O')
    final_color_vec = np.clip(final_color_vec, 0, 1)
    
    dot_color_actual = tuple(final_color_vec)
    label_font_color = 'black'
    if np.array_equal(final_color_vec, [1,1,1]): # White
        dot_color_actual = (0.85, 0.85, 0.85) # Light grey for dot
    elif len(behaviors_involved_abbr) == 1 and (is_C or is_O): # Pure Red or Pure Blue
        label_font_color = 'white'
    elif len(behaviors_involved_abbr) == 2 and is_C and is_O : # Magenta (C&O)
         label_font_color = 'white'

    x = random.uniform(0.05, 0.95)
    y = random.uniform(0.05, 0.95)
    neuron_attributes.append({
        'id': nid, 'x': x, 'y': y, 
        'dot_color': dot_color_actual, 
        'label_color': label_font_color,
        'behaviors_str': ' & '.join(behaviors_involved_abbr)
    })

# --- Generate Scatter Plot ---
fig, ax = plt.subplots(figsize=(14, 10)) # Increased figure size for legend
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

for attr in neuron_attributes:
    ax.scatter(attr['x'], attr['y'], color=attr['dot_color'], s=900, alpha=0.85, 
               edgecolors='grey' if attr['dot_color'] == (0.85,0.85,0.85) else 'none')
    ax.text(attr['x'], attr['y'], str(attr['id']), 
            ha='center', va='center', fontsize=9, color=attr['label_color'], weight='bold')

# --- Custom Legend: Overlapping Colors Diagram --- 
# Define colors and labels for the legend
legend_info = [
    ({'label': "Close (C)", 'color': tuple(color_map_rgb['Close'])}, ['C']),
    ({'label': "Middle (M)", 'color': tuple(color_map_rgb['Middle'])}, ['M']),
    ({'label': "Open (O)", 'color': tuple(color_map_rgb['Open'])}, ['O']),
    ({'label': "C & M", 'color': tuple(np.clip(color_map_rgb['Close'] + color_map_rgb['Middle'], 0, 1))}, ['C','M']),
    ({'label': "C & O", 'color': tuple(np.clip(color_map_rgb['Close'] + color_map_rgb['Open'], 0, 1))}, ['C','O']),
    ({'label': "M & O", 'color': tuple(np.clip(color_map_rgb['Middle'] + color_map_rgb['Open'], 0, 1))}, ['M','O']),
    ({'label': "C & M & O", 'color': (0.85, 0.85, 0.85)}, ['C','M','O'])
]

# Create a new axes for the legend on the right
legend_ax = fig.add_axes([0.82, 0.55, 0.15, 0.3]) # x, y, width, height (relative to figure)
legend_ax.set_title("Behavior Association", fontsize=10)
legend_ax.axis('off')

# Parameters for drawing circles in legend
circle_radius = 0.1
alpha_value = 0.7 # Transparency for base circles
label_offset = circle_radius + 0.03

# Positions for the three main circles (approximate for a Venn-like feel)
pos_C = (0.3, 0.65)
pos_M = (0.7, 0.65)
pos_O = (0.5, 0.3)

# Draw base circles with transparency
leg_c_circle = mpatches.Circle(pos_C, circle_radius, color=color_map_rgb['Close'], alpha=alpha_value, transform=legend_ax.transAxes)
leg_m_circle = mpatches.Circle(pos_M, circle_radius, color=color_map_rgb['Middle'], alpha=alpha_value, transform=legend_ax.transAxes)
leg_o_circle = mpatches.Circle(pos_O, circle_radius, color=color_map_rgb['Open'], alpha=alpha_value, transform=legend_ax.transAxes)
legend_ax.add_patch(leg_c_circle)
legend_ax.add_patch(leg_m_circle)
legend_ax.add_patch(leg_o_circle)

# Add small colored patches and text labels for each category
# These are positioned manually for a clear layout, not strict Venn intersections
legend_items_y_start = 0.95
legend_items_y_step = 0.13
legend_patch_size = 0.04

for i, (item, behaviors) in enumerate(legend_info):
    # Determine the actual color used in the plot for this combination
    current_behaviors_set = set(behaviors)
    actual_plot_color = item['color'] # Use the pre-defined mixed color
    if current_behaviors_set == {'C', 'M', 'O'}:
         actual_plot_color = (0.85, 0.85, 0.85) # Ensure it matches the dot color for CMO

    rect_y = legend_items_y_start - i * legend_items_y_step
    rect = mpatches.Rectangle((0.05, rect_y - legend_patch_size/2), legend_patch_size, legend_patch_size, 
                              facecolor=actual_plot_color, edgecolor='black', transform=legend_ax.transAxes)
    legend_ax.add_patch(rect)
    legend_ax.text(0.05 + legend_patch_size + 0.03, rect_y, item['label'], 
                   ha='left', va='center', fontsize=8, transform=legend_ax.transAxes)

legend_ax.set_ylim(0, 1) # Ensure all legend items fit
legend_ax.set_xlim(0, 1)

plt.suptitle(f'Top {top_n} Neuron Behavior Associations (EMtrace Data)\n(Colors indicate behavior associations - see legend)', fontsize=16, y=0.98) # Adjusted title
# fig.tight_layout(rect=[0, 0, 0.80, 0.95]) # Adjust rect to make space for legend and suptitle

output_filename = "neuron_color_scatter_emtrace_custom_legend.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"\nScatter plot with custom legend saved as: {output_filename}")
plt.show()

print("\nScript finished.") 