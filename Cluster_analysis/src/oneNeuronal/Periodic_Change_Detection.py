import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os
import re
import numpy as np
from scipy.signal import find_peaks
import seaborn as sns
from scipy import signal

# Set Matplotlib style and configure seaborn
sns.set_style("whitegrid")  # Use seaborn's whitegrid style
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300


def create_directory(path):
    """
    创建目录（如果不存在）
    """
    if not os.path.exists(path):
        os.makedirs(path)


def optimize_lag_via_fft(signal, sampling_interval, lags_range=(50, 500)):
    """
    Analyze signal spectrum via FFT and optimize lag range

    Parameters:
    - signal: Time series data (numpy array or pandas Series)
    - sampling_interval: Sampling interval (seconds)
    - lags_range: Lag range (tuple, e.g., (50, 500))

    Returns:
    - selected_lags: List of selected lags
    - selected_magnitudes: Peak magnitudes corresponding to selected lags
    - freq: All frequency components
    - fft_magnitude: Magnitudes for all frequencies
    - peak_freqs: Peak frequencies
    - peak_magnitudes: Peak magnitudes
    - is_periodic: Whether the signal has significant periodicity
    """
    # 1. Calculate FFT
    n = len(signal)
    freq = np.fft.rfftfreq(n, d=sampling_interval)  # Frequency
    fft_magnitude = np.abs(np.fft.rfft(signal))  # Magnitude

    # 2. Find peaks in spectrum
    prominence = np.mean(fft_magnitude) * 0.5  # Peak prominence threshold
    peaks, peak_properties = find_peaks(fft_magnitude, prominence=prominence)
    peak_freqs = freq[peaks]  # Corresponding frequencies
    peak_magnitudes = fft_magnitude[peaks]  # Peak magnitudes

    # 3. Determine periodicity
    is_periodic = False
    if len(peaks) > 0:
        max_magnitude = np.max(peak_magnitudes)
        background_magnitude = np.median(fft_magnitude)
        if max_magnitude > background_magnitude * 3:  # Peak at least 3x background
            is_periodic = True

    # 4. Calculate lags from frequencies
    peak_freqs_nonzero = peak_freqs[peak_freqs > 0]
    peak_magnitudes_nonzero = peak_magnitudes[peak_freqs > 0]
    lag_values = (1 / peak_freqs_nonzero) / sampling_interval

    print(f"Periodicity Detection Result: {'Significant Periodicity' if is_periodic else 'No Significant Periodicity'}")
    print("Key Frequencies and Corresponding Lags:")
    for f, lag, mag in zip(peak_freqs_nonzero, lag_values, peak_magnitudes_nonzero):
        print(f"Frequency: {f:.4f} Hz, Lag: {int(lag)}, Magnitude: {mag:.4f}")

    # 5. Select lags within range
    selected_lags = [int(lag) for lag in lag_values if lags_range[0] <= lag <= lags_range[1]]
    selected_magnitudes = [mag for lag, mag in zip(lag_values, peak_magnitudes_nonzero) if
                           lags_range[0] <= lag <= lags_range[1]]

    return selected_lags, selected_magnitudes, freq, fft_magnitude, peak_freqs, peak_magnitudes, is_periodic


def plot_comprehensive_analysis(calcium_series, time_step, neuron, sheet_output_dir):
    """
    Create comprehensive analysis plot including original signal, ACF, PACF, and FFT
    """
    # Close all existing figures
    plt.close('all')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 24))
    
    # 1. Original signal
    ax1 = plt.subplot(411)
    time_points = calcium_series.index  # Use actual timestamps
    ax1.plot(time_points, calcium_series, 'b-', linewidth=1)
    ax1.set_title(f'Original Calcium Signal - {neuron}')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Fluorescence Intensity')
    
    # 2. FFT analysis
    ax2 = plt.subplot(412)
    selected_lags, selected_magnitudes, freq, fft_magnitude, peak_freqs, peak_magnitudes, is_periodic = \
        optimize_lag_via_fft(calcium_series.values, time_step)
    
    ax2.plot(freq, fft_magnitude, 'b-', linewidth=1)
    ax2.set_title(f'Frequency Spectrum (FFT) - {("Periodic" if is_periodic else "Non-Periodic")}')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude')
    
    # Annotate major peaks
    for f, m in zip(peak_freqs, peak_magnitudes):
        if m > np.max(fft_magnitude) * 0.1:
            ax2.annotate(f'{f:.3f}Hz', 
                        xy=(f, m), 
                        xytext=(5, 5), 
                        textcoords='offset points')
    
    # 3. ACF
    ax3 = plt.subplot(413)
    plot_acf(calcium_series, lags=100, ax=ax3, title='Autocorrelation Function (ACF)')
    ax3.set_xlabel(f'Lag (steps of {time_step:.2f} seconds)')
    
    # 4. PACF
    ax4 = plt.subplot(414)
    plot_pacf(calcium_series, lags=100, ax=ax4, title='Partial Autocorrelation Function (PACF)', method='ywm')
    ax4.set_xlabel(f'Lag (steps of {time_step:.2f} seconds)')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    image_filename = f"{neuron}_comprehensive_analysis.png"
    image_path = os.path.join(sheet_output_dir, image_filename)
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return is_periodic


def main():
    # 1. Set input Excel file path
    excel_file = '../../datasets/processed_Day9.xlsx'

    # 2. Set output directory
    output_dir = '../../graph/periodic_analysis_day9'
    create_directory(output_dir)

    # 3. Read Excel file
    try:
        df = pd.read_excel(excel_file, engine='openpyxl')
    except Exception as e:
        print(f"Failed to read Excel file: {e}")
        return

    # 4. Process the data
    if df.shape[1] < 2:
        print("Insufficient columns in the data.")
        return

    # Use 'stamp' column for time information
    if 'stamp' not in df.columns:
        print("No 'stamp' column found in the data.")
        return

    # Ensure stamp column is numeric
    df['stamp'] = pd.to_numeric(df['stamp'], errors='coerce')

    # Check stamp column conversion success
    if df['stamp'].isnull().all():
        print("Stamp column conversion failed.")
        return

    # Handle missing values
    df = df.ffill().dropna()

    # Extract neuron names (all columns containing 'n')
    neuron_cols = [col for col in df.columns if 'n' in col.lower()]
    if not neuron_cols:
        print("No neuron columns found in the data.")
        return
    print(f"Found {len(neuron_cols)} neuron columns:", neuron_cols)

    # Record periodicity results
    periodic_results = {}

    # Process each neuron
    for neuron in neuron_cols:
        print(f"Processing neuron: {neuron}")

        # Extract time series and corresponding timestamps
        calcium_series = df[neuron]
        timestamps = df['stamp']

        # Check for sufficient data
        if calcium_series.empty:
            print(f"No data for neuron {neuron}, skipping.")
            continue

        # Calculate actual time step from timestamps
        actual_time_step = np.median(np.diff(timestamps))
        print(f"Actual time step: {actual_time_step:.4f} seconds")

        # Perform comprehensive analysis and get periodicity result
        is_periodic = plot_comprehensive_analysis(calcium_series, actual_time_step, neuron, output_dir)
        periodic_results[neuron] = is_periodic

    # Save periodicity analysis results
    results_df = pd.DataFrame.from_dict(periodic_results, orient='index', columns=['is_periodic'])
    results_df.to_excel(os.path.join(output_dir, 'periodic_analysis_results.xlsx'))

    # Plot periodicity distribution pie chart
    plt.figure(figsize=(8, 8))
    periodic_count = sum(periodic_results.values())
    non_periodic_count = len(periodic_results) - periodic_count
    plt.pie([periodic_count, non_periodic_count], 
            labels=['Periodic', 'Non-Periodic'],
            autopct='%1.1f%%',
            colors=['lightblue', 'lightcoral'])
    plt.title('Neuron Periodicity Distribution')
    plt.savefig(os.path.join(output_dir, 'periodic_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("Periodic analysis completed.")


if __name__ == "__main__":
    main()