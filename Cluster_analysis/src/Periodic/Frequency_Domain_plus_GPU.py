import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fftfreq
import pywt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os
import torch
import torch.fft
from torchaudio.functional import filtfilt

# Check if CUDA is available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

class FrequencyAnalyzer:
    def __init__(self, sampling_rate=10):
        """Initialize the FrequencyAnalyzer with sampling rate"""
        self.fs = sampling_rate
        self.delta_t = 1/sampling_rate

    def to_tensor(self, data):
        """Convert numpy array to PyTorch tensor on appropriate device"""
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(torch.float32).to(DEVICE)
        return data.to(torch.float32).to(DEVICE)

    def to_numpy(self, tensor):
        """Convert PyTorch tensor to numpy array"""
        if torch.is_tensor(tensor):
            return tensor.cpu().numpy()
        return tensor

    def load_data(self, file_path):
        """Load neuron time series data from Excel file"""
        df = pd.read_excel(file_path)
        # Select only neuron columns (n1 to n62)
        neuron_columns = [f'n{i}' for i in range(1, 63)]
        existing_neuron_cols = [col for col in neuron_columns if col in df.columns]
        return df[existing_neuron_cols]

    def perform_fft_analysis(self, signal_data):
        """Perform FFT analysis on the input signal using PyTorch"""
        # Convert to tensor
        signal_tensor = self.to_tensor(signal_data)
        n = len(signal_data)
        
        # Perform FFT
        fft_result = torch.fft.fft(signal_tensor)
        freqs = torch.from_numpy(fftfreq(n, self.delta_t)).to(DEVICE)
        
        # Get positive frequencies and magnitudes
        pos_mask = freqs >= 0
        freqs = freqs[pos_mask]
        magnitudes = torch.abs(fft_result[pos_mask])
        
        return self.to_numpy(freqs), self.to_numpy(magnitudes)

    def calculate_psd(self, signal_data):
        """Calculate Power Spectral Density using Welch's method"""
        # Since PyTorch doesn't have a direct Welch's method implementation,
        # we'll implement a simplified version
        signal_tensor = self.to_tensor(signal_data)
        nperseg = 1024
        noverlap = nperseg // 2
        
        # Split signal into segments
        step = nperseg - noverlap
        shape = ((signal_tensor.shape[-1] - noverlap) // step, nperseg)
        strides = (step * signal_tensor.stride(0), signal_tensor.stride(0))
        segments = torch.as_strided(signal_tensor, shape, strides)
        
        # Apply window function
        window = torch.hann_window(nperseg, device=DEVICE)
        segments = segments * window
        
        # Calculate periodogram
        scale = 1.0 / (self.fs * torch.sum(window**2))
        spec = torch.abs(torch.fft.rfft(segments, dim=1))**2
        spec = spec * scale
        
        # Average periodograms
        psd = torch.mean(spec, dim=0)
        
        # Generate frequency array
        freqs = torch.linspace(0, self.fs/2, psd.shape[0], device=DEVICE)
        
        return self.to_numpy(freqs), self.to_numpy(psd)

    def compute_spectrogram(self, signal_data):
        """Compute spectrogram using PyTorch"""
        signal_tensor = self.to_tensor(signal_data)
        nperseg = 256
        noverlap = 128
        
        # Split signal into segments
        step = nperseg - noverlap
        shape = ((signal_tensor.shape[-1] - noverlap) // step, nperseg)
        strides = (step * signal_tensor.stride(0), signal_tensor.stride(0))
        segments = torch.as_strided(signal_tensor, shape, strides)
        
        # Apply window
        window = torch.hann_window(nperseg, device=DEVICE)
        segments = segments * window
        
        # Compute FFT
        Sxx = torch.abs(torch.fft.rfft(segments, dim=1))**2
        
        # Generate time and frequency arrays
        times = torch.arange(0, segments.shape[0], device=DEVICE) * step / self.fs
        freqs = torch.linspace(0, self.fs/2, Sxx.shape[1], device=DEVICE)
        
        # Transpose Sxx to match the expected dimensions (freq x time)
        Sxx = Sxx.T
        
        return (self.to_numpy(freqs), self.to_numpy(times), 
                self.to_numpy(Sxx))

    def bandpass_filter(self, signal_data, lowcut, highcut, order=4):
        """Apply bandpass filter using PyTorch"""
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        
        # Ensure frequencies are within valid range
        low = np.clip(low, 0.001, 0.99)
        high = np.clip(high, 0.001, 0.99)
        
        if low >= high:
            low, high = high/2, high
        
        # Get filter coefficients
        b, a = signal.butter(order, [low, high], btype='band')
        # Convert to float32 tensors
        b = torch.tensor(b, dtype=torch.float32, device=DEVICE)
        a = torch.tensor(a, dtype=torch.float32, device=DEVICE)
        
        # Apply filter
        signal_tensor = self.to_tensor(signal_data)
        filtered = filtfilt(signal_tensor, a, b)
        
        return self.to_numpy(filtered)

    def calculate_phase(self, signal_data):
        """Calculate instantaneous phase using Hilbert transform with PyTorch"""
        # Normalize signal
        signal_tensor = self.to_tensor(signal_data)
        signal_normalized = (signal_tensor - torch.mean(signal_tensor)) / torch.std(signal_tensor)
        
        # Perform Hilbert transform using FFT
        n = len(signal_normalized)
        X = torch.fft.fft(signal_normalized)
        h = torch.zeros(n, device=DEVICE)
        if n % 2 == 0:
            h[0] = h[n//2] = 1
            h[1:n//2] = 2
        else:
            h[0] = 1
            h[1:(n+1)//2] = 2
            
        analytic_signal = torch.fft.ifft(X * h)
        amplitude_envelope = torch.abs(analytic_signal)
        instantaneous_phase = torch.angle(analytic_signal)
        
        return (self.to_numpy(amplitude_envelope), 
                self.to_numpy(instantaneous_phase))

    def extract_features(self, signal_data):
        """Extract comprehensive frequency domain features"""
        # Convert to tensor for GPU computation
        signal_tensor = self.to_tensor(signal_data)
        
        # FFT features
        freqs, magnitudes = self.perform_fft_analysis(signal_data)
        main_freq = freqs[torch.argmax(self.to_tensor(magnitudes)).item()]
        peak_power = torch.max(self.to_tensor(magnitudes)).item()
        total_power = torch.sum(self.to_tensor(magnitudes)).item()

        # Band power features
        theta = self.bandpass_filter(signal_data, 4, 8)
        gamma = self.bandpass_filter(signal_data, 30, min(100, self.fs/2 - 1))
        theta_power = float(torch.sum(self.to_tensor(theta)**2).item())
        gamma_power = float(torch.sum(self.to_tensor(gamma)**2).item())

        # Time domain features
        mean_val = float(torch.mean(signal_tensor).item())
        std_val = float(torch.std(signal_tensor).item())
        max_val = float(torch.max(signal_tensor).item())
        min_val = float(torch.min(signal_tensor).item())

        features = {
            'main_frequency': main_freq,
            'peak_power': peak_power,
            'total_power': total_power,
            'theta_power': theta_power,
            'gamma_power': gamma_power,
            'mean': mean_val,
            'std': std_val,
            'max': max_val,
            'min': min_val
        }
        return features

    def plot_comprehensive_analysis(self, neuron_id, time_series, output_dir):
        """Generate comprehensive frequency analysis plots"""
        try:
            # Create figure with subplots
            fig = plt.figure(figsize=(15, 20))
            
            # 1. Original Signal
            ax1 = plt.subplot(611)
            time = np.arange(len(time_series))/self.fs
            ax1.plot(time, time_series)
            ax1.set_title(f'Original Signal - {neuron_id}')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Amplitude')
            
            # 2. FFT
            ax2 = plt.subplot(612)
            freqs, magnitudes = self.perform_fft_analysis(time_series)
            ax2.plot(freqs, magnitudes)
            ax2.set_title('Frequency Spectrum (FFT)')
            ax2.set_xlabel('Frequency (Hz)')
            ax2.set_ylabel('Magnitude')
            ax2.set_xlim(0, self.fs/2)
            
            # 3. PSD
            ax3 = plt.subplot(613)
            freqs_psd, psd = self.calculate_psd(time_series)
            ax3.semilogy(freqs_psd, psd)
            ax3.set_title('Power Spectral Density')
            ax3.set_xlabel('Frequency (Hz)')
            ax3.set_ylabel('Power/Frequency')
            ax3.set_xlim(0, self.fs/2)
            
            # 4. Spectrogram
            ax4 = plt.subplot(614)
            freqs_spec, times_spec, Sxx = self.compute_spectrogram(time_series)
            # Add small constant to avoid log(0)
            Sxx = Sxx + 1e-10
            im = ax4.pcolormesh(times_spec, freqs_spec, 10 * np.log10(Sxx), 
                               shading='gouraud', cmap='viridis')
            ax4.set_title('Spectrogram')
            ax4.set_ylabel('Frequency (Hz)')
            ax4.set_xlabel('Time (s)')
            plt.colorbar(im, ax=ax4, label='Power/Frequency (dB/Hz)')
            
            # 5. Filtered Signals
            ax5 = plt.subplot(615)
            theta = self.bandpass_filter(time_series, 4, 8)
            gamma = self.bandpass_filter(time_series, 30, min(100, self.fs/2 - 1))
            ax5.plot(time, theta, label='Theta (4-8 Hz)')
            ax5.plot(time, gamma, label=f'Gamma (30-{min(100, self.fs/2 - 1)} Hz)')
            ax5.set_title('Filtered Signals')
            ax5.set_xlabel('Time (s)')
            ax5.set_ylabel('Amplitude')
            ax5.legend()
            
            # 6. Phase Analysis
            ax6 = plt.subplot(616)
            amp_env, inst_phase = self.calculate_phase(time_series)
            ax6.plot(time, inst_phase)
            ax6.set_title('Instantaneous Phase')
            ax6.set_xlabel('Time (s)')
            ax6.set_ylabel('Phase (radians)')
            
            plt.tight_layout()
            
            # Save the figure
            plt.savefig(os.path.join(output_dir, f'comprehensive_analysis_{neuron_id}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"Error processing {neuron_id}: {str(e)}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            plt.close('all')

def main():
    # Initialize analyzer
    analyzer = FrequencyAnalyzer(sampling_rate=10)
    
    # Set file paths
    file_path = '../../datasets/processed_Day6.xlsx'
    output_dir = '../../graph/frequency_analysis_day6_extended'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    df = analyzer.load_data(file_path)
    
    # Process each neuron
    feature_data = []
    for neuron_id in df.columns:
        print(f"Analyzing {neuron_id}...")
        time_series = df[neuron_id].values
        
        # Perform comprehensive analysis and generate plots
        analyzer.plot_comprehensive_analysis(neuron_id, time_series, output_dir)
        
        # Extract features
        features = analyzer.extract_features(time_series)
        features['neuron_id'] = neuron_id
        feature_data.append(features)
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Create and save features DataFrame
    features_df = pd.DataFrame(feature_data)
    features_df.to_excel(os.path.join('../../datasets/frequency_features_day6.xlsx'), index=False)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()
