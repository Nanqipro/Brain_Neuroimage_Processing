import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import seaborn as sns
import os

def load_data(file_path):
    """Load neuron time series data from Excel file"""
    df = pd.read_excel(file_path)
    # Select only neuron columns (n1 to n62)
    neuron_columns = [f'n{i}' for i in range(1, 63)]
    existing_neuron_cols = [col for col in neuron_columns if col in df.columns]
    return df[existing_neuron_cols]

def perform_fft_analysis(signal_data, sampling_rate):
    """Perform FFT analysis on the input signal"""
    n = len(signal_data)
    fft_result = fft(signal_data)
    freqs = fftfreq(n, 1/sampling_rate)
    
    # Get the positive frequencies and their magnitudes
    pos_mask = freqs >= 0
    freqs = freqs[pos_mask]
    magnitudes = np.abs(fft_result)[pos_mask]
    
    return freqs, magnitudes

def calculate_psd(signal_data, sampling_rate):
    """Calculate Power Spectral Density using Welch's method"""
    freqs, psd = signal.welch(signal_data, fs=sampling_rate)
    return freqs, psd

def plot_frequency_analysis(neuron_id, time_series, sampling_rate):
    """Plot the original signal, FFT result, and PSD"""
    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot original signal
    time = np.arange(len(time_series))/sampling_rate
    ax1.plot(time, time_series)
    ax1.set_title(f'Original Signal - {neuron_id}')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    
    # Plot FFT
    freqs, magnitudes = perform_fft_analysis(time_series, sampling_rate)
    ax2.plot(freqs, magnitudes)
    ax2.set_title('Frequency Spectrum (FFT)')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude')
    ax2.set_xlim(0, sampling_rate/2)  # Nyquist frequency
    
    # Plot PSD
    freqs_psd, psd = calculate_psd(time_series, sampling_rate)
    ax3.semilogy(freqs_psd, psd)
    ax3.set_title('Power Spectral Density')
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Power/Frequency')
    ax3.set_xlim(0, sampling_rate/2)
    
    plt.tight_layout()
    return fig

def main():
    # Get the absolute path to the workspace root
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    
    # Parameters
    file_path = '../../datasets/processed_Day3.xlsx'
    sampling_rate = 10  # Hz (assuming 10 Hz sampling rate, adjust if different)
    
    # Load data
    print("Loading data...")
    df = load_data(file_path)
    
    # Create output directory if it doesn't exist
    output_dir = '../../graph/frequency_analysis_day3'
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze each neuron
    for neuron_id in df.columns:
        print(f"Analyzing {neuron_id}...")
        time_series = df[neuron_id].values
        
        # Perform frequency analysis and create plots
        fig = plot_frequency_analysis(neuron_id, time_series, sampling_rate)
        
        # Save the figure
        fig.savefig(f'{output_dir}/frequency_analysis_{neuron_id}.png')
        plt.close(fig)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()
