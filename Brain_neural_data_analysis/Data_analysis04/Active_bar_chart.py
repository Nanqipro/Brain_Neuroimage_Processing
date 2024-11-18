# # 无行为标注
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
#
# # Load the dataset
# file_path = './data/Day6_Neuron_Calcium_Metrics.csv'  # 替换为实际路径
# metrics_df = pd.read_csv(file_path)
#
# # Ensure 'Neuron', 'Start Time', and 'GMM' columns exist
# if 'Neuron' not in metrics_df.columns or 'Start Time' not in metrics_df.columns or 'GMM' not in metrics_df.columns:
#     raise ValueError("The dataset must contain 'Neuron', 'Start Time', and 'GMM' columns.")
#
# # Map neurons to sequential IDs based on their order in the dataset
# metrics_df['Neuron'] = metrics_df['Neuron'].astype(str)  # Ensure neuron labels are strings
# unique_neurons = metrics_df['Neuron'].drop_duplicates().tolist()  # Preserve original order
# neuron_mapping = {neuron: i + 1 for i, neuron in enumerate(unique_neurons)}
# metrics_df['Neuron_Mapped'] = metrics_df['Neuron'].map(neuron_mapping)
#
# # Configure plotting
# plt.figure(figsize=(20, 15))
#
# # Use Set3 colormap for cluster colors
# colors = plt.cm.Set3.colors  # Replace this with any other colormap as desired
# # colors = plt.cm.tab10.colors
# # colors = plt.cm.Paired.colors
#
#
# # Generate a legend for clusters
# unique_clusters = sorted(metrics_df['GMM'].unique())
# handles = [
#     mpatches.Patch(color=colors[int(cluster) % len(colors)], label=f"Cluster {int(cluster)}")
#     for cluster in unique_clusters
# ]
#
# # Plot each neuron as a horizontal line with color changes based on clusters
# for neuron in unique_neurons:
#     neuron_data = metrics_df[metrics_df['Neuron'] == neuron]
#     y_position = neuron_mapping[neuron]
#
#     for _, row in neuron_data.iterrows():
#         start_time = row['Start Time']
#         end_time = start_time + 5  # Step size is 5 based on the instruction
#         cluster = int(row['GMM'])
#         plt.plot(
#             [start_time, end_time],
#             [y_position, y_position],
#             color=colors[cluster % len(colors)],
#             linewidth=5,
#         )
#
# # Add labels and legend
# plt.xlabel('Time (stamp)', fontsize=16, fontweight='bold')  # 加粗并放大横坐标字体
# plt.ylabel('Neuron ID', fontsize=16, fontweight='bold')     # 加粗并放大纵坐标字体
# plt.title('Time Activity Raster Plot of Neurons by Cluster (GMM)', fontsize=18, fontweight='bold')  # 加粗标题
# plt.xticks(fontsize=14, fontweight='bold')  # 调整横坐标刻度字体大小和加粗
# plt.yticks(range(1, len(unique_neurons) + 1), unique_neurons, fontsize=12, fontweight='bold')  # 调整纵坐标刻度
# plt.legend(handles=handles, title="Clusters", fontsize=12, title_fontsize=14, loc='upper right')
# plt.grid(axis='x', linestyle='--', alpha=0.7)
# plt.tight_layout()
#
#
# # Save the plot
# # output_image_path = './data/Time_Activity_Raster_Plot_GMM_Set3.png'
# # plt.savefig(output_image_path)
# plt.show()
#
# # print(f"图像已保存至: {output_image_path}")
#


# 行为学标注
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Load the dataset
file_path = './data/Day6_Neuron_Calcium_Metrics.xlsx'  # 替换为实际路径
metrics_df = pd.read_excel(file_path, sheet_name='Windows100_step10')

# Ensure 'Neuron', 'Start Time', and 'k-means-Manhattan' columns exist
if 'Neuron' not in metrics_df.columns or 'Start Time' not in metrics_df.columns or 'GMM' not in metrics_df.columns:
    raise ValueError("The dataset must contain 'Neuron', 'Start Time', and 'GMM' columns.")

# Map neurons to sequential IDs based on their order in the dataset
metrics_df['Neuron'] = metrics_df['Neuron'].astype(str)  # Ensure neuron labels are strings
unique_neurons = metrics_df['Neuron'].drop_duplicates().tolist()  # Preserve original order
neuron_mapping = {neuron: i + 1 for i, neuron in enumerate(unique_neurons)}
metrics_df['Neuron_Mapped'] = metrics_df['Neuron'].map(neuron_mapping)

# Load behavioral data (assuming it's a separate Excel file and from 'CHB' sheet)
behavioral_file_path = './data/day6cell行为学标注.xlsx'  # 替换为实际路径
behavioral_df = pd.read_excel(behavioral_file_path, sheet_name='CHB')  # 指定工作表为 'CHB'

# 打印列名查看
print(behavioral_df.columns)

# Ensure the 'Behavioral' column exists
if 'Behavioral' not in behavioral_df.columns or 'stamp' not in behavioral_df.columns:
    raise ValueError("The dataset must contain 'Behavioral' and 'stamp' columns.")

# Configure plotting
plt.figure(figsize=(30, 15))

# Use Set3 colormap for cluster colors
# colors = plt.cm.Set3.colors  # Replace this with any other colormap as desired
# colors = plt.cm.tab10.colors
colors = plt.cm.Paired.colors

# Generate a legend for clusters
unique_clusters = sorted(metrics_df['GMM'].unique())
handles = [
    mpatches.Patch(color=colors[int(cluster) % len(colors)], label=f"Cluster {int(cluster)}")
    for cluster in unique_clusters
]

# Plot each neuron as a horizontal line with color changes based on clusters
for neuron in unique_neurons:
    neuron_data = metrics_df[metrics_df['Neuron'] == neuron]
    y_position = neuron_mapping[neuron]

    for _, row in neuron_data.iterrows():
        start_time = row['Start Time']
        end_time = start_time + 5  # Step size is 5 based on the instruction
        cluster = int(row['GMM'])
        plt.plot(
            [start_time, end_time],
            [y_position, y_position],
            color=colors[cluster % len(colors)],
            linewidth=5,
        )

# Add behavioral annotations (white line and label)
for _, behavioral_row in behavioral_df[behavioral_df['Behavioral'].notna()].iterrows():
    stamp = behavioral_row['stamp']
    behavior = behavioral_row['Behavioral']

    # Plot a black vertical line at the behavioral 'stamp'
    plt.axvline(x=stamp, color='black', linewidth=1, linestyle='--')

    # Add behavioral annotation next to the line (on the right side)
    plt.text(stamp + 0.5, len(unique_neurons) + 1, behavior, color='black', ha='left', va='bottom', fontsize=12, rotation=90)

# Add labels and legend
plt.xlabel('Time (stamp)', fontsize=16, fontweight='bold')  # 加粗并放大横坐标字体
plt.ylabel('Neuron ID', fontsize=16, fontweight='bold')  # 加粗并放大纵坐标字体
plt.title('Time Activity Raster Plot of Neurons by Cluster (GMM)', fontsize=18, fontweight='bold')  # 加粗标题
plt.xticks(fontsize=14, fontweight='bold')  # 调整横坐标刻度字体大小和加粗
plt.yticks(range(1, len(unique_neurons) + 1), unique_neurons, fontsize=12, fontweight='bold')  # 调整纵坐标刻度
plt.legend(handles=handles, title="Clusters", fontsize=12, title_fontsize=14, loc='upper right')
# plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the plot
output_image_path = './data/GMM.png'
plt.savefig(output_image_path)
plt.show()

print(f"图像已保存至: {output_image_path}")
