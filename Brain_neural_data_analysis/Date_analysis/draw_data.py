import pandas as pd
import matplotlib.pyplot as plt

# Load the Excel file
# file_path = './data/smoothed_trace_homecage.xlsx'
file_path = './data/2979 CSDS Day3.xlsx'
df = pd.read_excel(file_path)

# Set up the plot
plt.figure(figsize=(10, 8))

# Scaling factor to make the amplitude more visible
scaling_factor = 30

# Plot each line at a different vertical level, with scaling applied
for i in range(1, 53):
    plt.plot(df['stamp'], df[f'n{i}'] * scaling_factor + i * 30, label=f'n{i}')  # Scaling and vertical stacking

# Set x-axis limits to 0 to 500
plt.xlim(0,3000)

# Add vertical dashed lines and text annotations (adjusting to fit the new scale)
for x_pos in []:
    plt.axvline(x=x_pos, color='red', linestyle='--', linewidth=0.8)  # Red vertical lines
    plt.axvline(x=x_pos - 50, color='blue', linestyle='--', linewidth=0.8)  # Blue vertical lines

# # Text annotations
# plt.text(100, 1050, 'enclosed arms', ha='center', fontsize=12)
# plt.text(400, 1050, 'open arms', ha='center', fontsize=12)

# Adding labels and title
plt.xlabel('Stamp')
plt.ylabel('Traces (n1 ~ n53)')
plt.title('Traces with Increased Amplitude')

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
