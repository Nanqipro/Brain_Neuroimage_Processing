## 无标准化
# import numpy as np
# from scipy.signal import savgol_filter
# import pandas as pd
#
#
# # Step 1.2: Apply filtering to remove noise and smooth the signal
# # Define a function to apply Savitzky-Golay filter to the ΔF/F data
#
# def smooth_signal(data, window_length=11, polyorder=2):
#     """
#     Applies a Savitzky-Golay filter to smooth the data.
#
#     Args:
#     - data (DataFrame): The ΔF/F data with rows as time points and columns as neurons.
#     - window_length (int): The length of the filter window (odd integer).
#     - polyorder (int): The order of the polynomial used to fit the samples.
#
#     Returns:
#     - DataFrame: Smoothed ΔF/F data.
#     """
#     smoothed_data = data.copy()
#     for column in smoothed_data.columns[1:]:  # Skip the 'stamp' column
#         smoothed_data[column] = savgol_filter(smoothed_data[column], window_length, polyorder)
#     return smoothed_data
#
#
# # Load your original data (make sure to adjust the file path as necessary)
# file_path = '/data/trace_homecage.xlsx'
# data = pd.read_excel(file_path)
#
# # Apply the smoothing function to the ΔF/F data
# smoothed_data = smooth_signal(data)
#
# # Specify the file path where you want to save the smoothed data
# output_file_path = '/data/smoothed_trace_homecage.xlsx'
#
# # Save the smoothed data to an Excel file
# smoothed_data.to_excel(output_file_path, index=False)
#
# # Confirm the data is saved
# print(f"Smoothed data has been saved to: {output_file_path}")

# 有Z-score标准化
import numpy as np
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Step 1: Apply filtering to remove noise and smooth the signal
def smooth_signal(data, window_length=11, polyorder=2):
    """
    Applies a Savitzky-Golay filter to smooth the data.

    Args:
    - data (DataFrame): The ΔF/F data with rows as time points and columns as neurons.
    - window_length (int): The length of the filter window (odd integer).
    - polyorder (int): The order of the polynomial used to fit the samples.

    Returns:
    - DataFrame: Smoothed ΔF/F data.
    """
    smoothed_data = data.copy()
    for column in smoothed_data.columns[1:]:  # Skip the 'stamp' column
        smoothed_data[column] = savgol_filter(smoothed_data[column], window_length, polyorder)
    return smoothed_data

# Step 2: Apply Z-score normalization (Z-score scaling)
def zscore_normalize(data):
    """
    Applies Z-score normalization to the data.
    Args:
    - data (DataFrame): The ΔF/F data with rows as time points and columns as neurons.
    Returns:
    - DataFrame: Z-score normalized data.
    """
    scaler = StandardScaler()
    zscore_data = data.copy()
    zscore_data.iloc[:, 1:] = scaler.fit_transform(data.iloc[:, 1:])  # Skip the 'stamp' column
    return zscore_data

# Load your original data (make sure to adjust the file path as necessary)
file_path = './data/2979csds Day6.xlsx'
data = pd.read_excel(file_path)

# Step 1: Apply the smoothing function to the ΔF/F data
smoothed_data = smooth_signal(data)

# Step 2: Apply Z-score normalization
normalized_data = zscore_normalize(smoothed_data)

# Specify the file path where you want to save the smoothed and normalized data
output_file_path = './data/smoothed_normalized_2979_CSDS_Day6.xlsx'

# Save the smoothed and normalized data to an Excel file
normalized_data.to_excel(output_file_path, index=False)

# Confirm the data is saved
print(f"Smoothed and normalized data has been saved to: {output_file_path}")
