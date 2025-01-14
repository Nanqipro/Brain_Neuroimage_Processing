import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.signal import welch, spectrogram, butter, filtfilt, coherence, hilbert
import pywt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os

class FrequencyAnalyzer:
    def __init__(self, sampling_rate=10):
        """Initialize the FrequencyAnalyzer with sampling rate"""
        self.fs = sampling_rate
        self.delta_t = 1/sampling_rate

    def load_data(self, file_path, start_time=None, end_time=None):
        """
        Load neuron time series data from Excel file with optional time window selection
        
        Parameters:
        -----------
        file_path : str
            Path to the Excel file
        start_time : float, optional
            Start time in seconds for analysis window
        end_time : float, optional
            End time in seconds for analysis window
        """
        df = pd.read_excel(file_path)
        
        # Select time window if specified
        if 'stamp' in df.columns:
            if start_time is not None or end_time is not None:
                start_time = start_time if start_time is not None else df['stamp'].min()
                end_time = end_time if end_time is not None else df['stamp'].max()
                df = df[(df['stamp'] >= start_time) & (df['stamp'] <= end_time)].copy()
        
        # 获取所有包含'n'的列（不区分大小写）
        neuron_columns = [col for col in df.columns if 'n' in col.lower()]
        if not neuron_columns:
            raise ValueError("No neuron columns found in the Excel file!")
        print(f"Found {len(neuron_columns)} neuron columns:", neuron_columns)
        
        # Store time information
        self.timestamps = df['stamp'].values if 'stamp' in df.columns else None
        
        return df[neuron_columns]

    def perform_fft_analysis(self, signal_data):
        """Perform FFT analysis on the input signal"""
        n = len(signal_data)
        fft_result = fft(signal_data)
        freqs = fftfreq(n, self.delta_t)
        
        # Get the positive frequencies and their magnitudes
        pos_mask = freqs >= 0
        freqs = freqs[pos_mask]
        magnitudes = np.abs(fft_result)[pos_mask]
        
        return freqs, magnitudes

    def calculate_psd(self, signal_data):
        """Calculate Power Spectral Density using Welch's method"""
        # Adjust nperseg based on signal length
        nperseg = min(1024, len(signal_data))
        if nperseg < 4:  # Ensure minimum segment length
            nperseg = len(signal_data)
        noverlap = nperseg // 2
        freqs, psd = welch(signal_data, fs=self.fs, nperseg=nperseg, noverlap=noverlap)
        return freqs, psd

    def compute_spectrogram(self, signal_data):
        """Compute spectrogram of the signal"""
        # Adjust nperseg based on signal length
        nperseg = min(256, len(signal_data) // 4)  # Ensure at least 4 segments
        if nperseg < 4:  # Ensure minimum segment length
            nperseg = len(signal_data)
        noverlap = min(128, nperseg // 2)  # Adjust overlap accordingly
        
        freqs, times, Sxx = spectrogram(signal_data, fs=self.fs, 
                                      nperseg=nperseg, noverlap=noverlap)
        return freqs, times, Sxx

    def perform_wavelet_analysis(self, signal_data, wavelet='db4', level=5):
        """Perform wavelet decomposition"""
        coeffs = pywt.wavedec(signal_data, wavelet, level=level)
        return coeffs

    def bandpass_filter(self, signal_data, lowcut, highcut, order=4):
        """Apply bandpass filter to the signal"""
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        
        # Ensure frequencies are within valid range (0 < freq < 1)
        low = np.clip(low, 0.001, 0.99)
        high = np.clip(high, 0.001, 0.99)
        
        # Ensure low < high
        if low >= high:
            low, high = high/2, high
        
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, signal_data)
        return y

    def calculate_phase(self, signal_data):
        """Calculate instantaneous phase using Hilbert transform"""
        # Normalize signal before Hilbert transform
        signal_normalized = (signal_data - np.mean(signal_data)) / np.std(signal_data)
        analytic_signal = hilbert(signal_normalized)
        amplitude_envelope = np.abs(analytic_signal)
        instantaneous_phase = np.angle(analytic_signal)
        return amplitude_envelope, instantaneous_phase

    def calculate_coherence(self, signal1, signal2):
        """Calculate coherence between two signals"""
        freqs, coh = coherence(signal1, signal2, fs=self.fs, nperseg=1024)
        return freqs, coh

    def calculate_autocorrelation(self, signal_data, max_lags=None):
        """
        Calculate autocorrelation of the signal
        
        Parameters:
        -----------
        signal_data : array-like
            Input signal
        max_lags : int, optional
            Maximum number of lags to compute. If None, uses min(len(signal_data)-1, 1000)
        
        Returns:
        --------
        lags : array
            Lag values in seconds
        acf : array
            Autocorrelation values
        """
        try:
            # Convert input to numpy array and ensure it's 1D
            signal_data = np.asarray(signal_data).ravel()
            
            # Remove any NaN values
            signal_data = signal_data[~np.isnan(signal_data)]
            
            # Normalize signal
            signal_norm = (signal_data - np.mean(signal_data)) / (np.std(signal_data) + 1e-10)
            
            # Set maximum lags
            if max_lags is None:
                max_lags = min(len(signal_data)-1, 1000)  # Limit to 1000 lags for efficiency
            
            # Calculate autocorrelation using numpy correlate
            acf = np.correlate(signal_norm, signal_norm, mode='full')
            
            # Keep only positive lags and normalize
            n = len(signal_data)
            acf = acf[n-1:] / acf[n-1]  # Normalize by zero-lag value
            acf = acf[:max_lags+1]      # Limit to max_lags
            
            # Convert lags to time
            lags = np.arange(len(acf)) / self.fs
            
            return lags, acf
            
        except Exception as e:
            print(f"Error in autocorrelation calculation: {str(e)}")
            raise

    def find_periodic_components(self, acf, lags, threshold=0.2):
        """
        Find periodic components from autocorrelation function
        
        Parameters:
        -----------
        acf : array
            Autocorrelation values
        lags : array
            Lag values in seconds
        threshold : float
            Minimum peak height to consider
        
        Returns:
        --------
        periods : list
            List of detected periods in seconds
        """
        try:
            from scipy.signal import find_peaks
            
            # Ensure arrays are 1D
            acf = np.asarray(acf).ravel()
            lags = np.asarray(lags).ravel()
            
            # Find peaks in autocorrelation
            # Add distance parameter to avoid detecting peaks too close to each other
            min_distance = int(0.5 * self.fs)  # Minimum 0.5 second between peaks
            peaks, _ = find_peaks(acf, height=threshold, distance=min_distance)
            
            if len(peaks) == 0:
                return []
            
            # Convert peak positions to periods
            periods = lags[peaks]
            
            # Sort periods by peak height
            peak_heights = acf[peaks]
            sorted_indices = np.argsort(peak_heights)[::-1]  # Sort in descending order
            periods = periods[sorted_indices]
            
            return periods.tolist()  # Convert to list for better compatibility
            
        except Exception as e:
            print(f"Error in finding periodic components: {str(e)}")
            return []

    def extract_features(self, signal_data):
        """Extract comprehensive frequency domain features"""
        try:
            # FFT features
            freqs, magnitudes = self.perform_fft_analysis(signal_data)
            main_freq = float(freqs[np.argmax(magnitudes)])
            peak_power = float(np.max(magnitudes))
            total_power = float(np.sum(magnitudes))

            # Band power features
            theta = self.bandpass_filter(signal_data, 4, 8)
            gamma = self.bandpass_filter(signal_data, 30, min(100, self.fs/2 - 1))
            theta_power = float(np.sum(theta**2))
            gamma_power = float(np.sum(gamma**2))

            # Autocorrelation features
            lags, acf = self.calculate_autocorrelation(signal_data)
            periods = self.find_periodic_components(acf, lags)
            
            # Get main period (if any)
            main_period = float(periods[0]) if periods else 0.0
            
            # Calculate ACF decay time (time to reach 1/e of initial value)
            decay_threshold = np.exp(-1)
            decay_indices = np.where(acf < decay_threshold)[0]
            decay_time = float(lags[decay_indices[0]]) if len(decay_indices) > 0 else float(lags[-1])

            # Time domain features
            mean_val = float(np.mean(signal_data))
            std_val = float(np.std(signal_data))
            max_val = float(np.max(signal_data))
            min_val = float(np.min(signal_data))

            features = {
                'main_frequency': main_freq,
                'peak_power': peak_power,
                'total_power': total_power,
                'theta_power': theta_power,
                'gamma_power': gamma_power,
                'main_period': main_period,
                'acf_decay_time': decay_time,
                'n_periods': len(periods),
                'mean': mean_val,
                'std': std_val,
                'max': max_val,
                'min': min_val
            }
            return features
        except Exception as e:
            print(f"Error in feature extraction: {str(e)}")
            raise

    def plot_comprehensive_analysis(self, neuron_id, time_series, output_dir):
        """Generate comprehensive frequency analysis plots"""
        try:
            # Create figure with subplots
            plt.close('all')  # Close all existing figures
            fig = plt.figure(figsize=(15, 24))  # Increased figure height for autocorrelation plot
            
            # 1. Original Signal
            ax1 = plt.subplot(711)  # Changed to 7 subplots
            # Use actual timestamps if available, otherwise use indices
            if self.timestamps is not None:
                time = self.timestamps
                ax1.set_xlabel('Time (s)')
            else:
                time = np.arange(len(time_series))/self.fs
                ax1.set_xlabel('Time (s)')
            
            ax1.plot(time, time_series)
            ax1.set_title(f'Original Signal - {neuron_id}')
            ax1.set_ylabel('Amplitude')
            
            # 2. FFT
            ax2 = plt.subplot(712)
            freqs, magnitudes = self.perform_fft_analysis(time_series)
            ax2.plot(freqs, magnitudes)
            ax2.set_title('Frequency Spectrum (FFT)')
            ax2.set_xlabel('Frequency (Hz)')
            ax2.set_ylabel('Magnitude')
            ax2.set_xlim(0, self.fs/2)
            
            # 3. PSD
            ax3 = plt.subplot(713)
            freqs_psd, psd = self.calculate_psd(time_series)
            ax3.semilogy(freqs_psd, psd)
            ax3.set_title('Power Spectral Density')
            ax3.set_xlabel('Frequency (Hz)')
            ax3.set_ylabel('Power/Frequency')
            ax3.set_xlim(0, self.fs/2)
            
            # 4. Spectrogram
            ax4 = plt.subplot(714)
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
            ax5 = plt.subplot(715)
            theta = self.bandpass_filter(time_series, 4, 8)
            gamma = self.bandpass_filter(time_series, 30, min(100, self.fs/2 - 1))
            ax5.plot(time, theta, label='Theta (4-8 Hz)')
            ax5.plot(time, gamma, label=f'Gamma (30-{min(100, self.fs/2 - 1)} Hz)')
            ax5.set_title('Filtered Signals')
            ax5.set_xlabel('Time (s)')
            ax5.set_ylabel('Amplitude')
            ax5.legend()
            
            # 6. Phase Analysis
            ax6 = plt.subplot(716)
            amp_env, inst_phase = self.calculate_phase(time_series)
            ax6.plot(time, inst_phase)
            ax6.set_title('Instantaneous Phase')
            ax6.set_xlabel('Time (s)')
            ax6.set_ylabel('Phase (radians)')
            
            # 7. Autocorrelation Analysis (New)
            ax7 = plt.subplot(717)
            lags, acf = self.calculate_autocorrelation(time_series)
            ax7.plot(lags, acf)
            ax7.set_title('Autocorrelation Function')
            ax7.set_xlabel('Lag (s)')
            ax7.set_ylabel('Correlation')
            ax7.grid(True)
            
            # Add detected periods
            periods = self.find_periodic_components(acf, lags)
            if periods:
                for period in periods:
                    ax7.axvline(x=period, color='r', linestyle='--', alpha=0.5)
                ax7.text(0.02, 0.98, f'Detected periods: {", ".join([f"{p:.1f}s" for p in periods])}',
                        transform=ax7.transAxes, verticalalignment='top')
            
            plt.tight_layout()
            
            # Add time window information to title if using specific window
            if self.timestamps is not None:
                window_info = f" (Time: {self.timestamps[0]:.1f}s - {self.timestamps[-1]:.1f}s)"
                fig.suptitle(f"Frequency Analysis for {neuron_id}{window_info}", y=1.02, fontsize=14)
            
            # Save the figure
            plt.savefig(os.path.join(output_dir, f'comprehensive_analysis_{neuron_id}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close(fig)  # Explicitly close the figure
            plt.close('all')  # Close any remaining figures
        except Exception as e:
            plt.close('all')  # Ensure figures are closed even if there's an error
            raise

def get_behavior_segments(df):
    """
    Get time segments for each continuous behavior period
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'stamp' and 'behavior' columns
    
    Returns:
    --------
    dict : Dictionary with behavior types as keys and list of (start_time, end_time) tuples as values
    """
    segments = {}
    
    # Get behavior changes
    behavior_changes = df['behavior'] != df['behavior'].shift()
    change_indices = df.index[behavior_changes].tolist()
    
    # Add start and end indices
    change_indices = [0] + change_indices + [len(df)]
    
    # Group segments by behavior
    for i in range(len(change_indices)-1):
        start_idx = change_indices[i]
        end_idx = change_indices[i+1]
        
        behavior = df.iloc[start_idx]['behavior']
        start_time = df.iloc[start_idx]['stamp']
        end_time = df.iloc[end_idx-1]['stamp']
        
        # Initialize list for this behavior if not exists
        if behavior not in segments:
            segments[behavior] = []
        
        # Add segment if it's long enough (at least 5 seconds)
        if end_time - start_time >= 5:
            segments[behavior].append((start_time, end_time))
    
    return segments

def main():
    # Initialize analyzer
    analyzer = FrequencyAnalyzer(sampling_rate=10)
    
    # Set file paths
    file_path = '../../datasets/processed_Day3.xlsx'
    output_base_dir = '../../graph/frequency_analysis_day3_behavior'
    
    # Load full data
    print("Loading dataset...")
    full_df = pd.read_excel(file_path)
    
    if 'behavior' not in full_df.columns:
        print("Error: No behavior column found in the dataset")
        return
    
    # Get behavior segments
    behavior_segments = get_behavior_segments(full_df)
    
    # Process each behavior type
    for behavior, segments in behavior_segments.items():
        print(f"\nAnalyzing behavior: {behavior}")
        
        # Create behavior-specific output directory
        behavior_dir = os.path.join(output_base_dir, f'behavior_{behavior}')
        os.makedirs(behavior_dir, exist_ok=True)
        
        # Save segment information
        segment_info = pd.DataFrame(segments, columns=['start_time', 'end_time'])
        segment_info['duration'] = segment_info['end_time'] - segment_info['start_time']
        segment_info.to_excel(os.path.join(behavior_dir, 'segments_info.xlsx'), index=False)
        
        # Process each segment for this behavior
        for segment_idx, (start_time, end_time) in enumerate(segments):
            try:
                # Create segment-specific directory
                segment_dir = os.path.join(behavior_dir, f'segment_{segment_idx+1:03d}')
                os.makedirs(segment_dir, exist_ok=True)
                
                print(f"  Analyzing segment {segment_idx+1}: {start_time:.1f}s - {end_time:.1f}s")
                
                # Load data for this segment
                df = analyzer.load_data(file_path, start_time=start_time, end_time=end_time)
                
                if len(df) < 50:  # Skip segments with too few samples
                    print(f"  Skipping segment: insufficient samples")
                    continue
                
                # Process each neuron
                feature_data = []
                for neuron_id in df.columns:
                    try:
                        print(f"    Analyzing {neuron_id}...")
                        time_series = df[neuron_id].values
                        
                        # Perform comprehensive analysis and generate plots
                        analyzer.plot_comprehensive_analysis(neuron_id, time_series, segment_dir)
                        
                        # Extract features
                        features = analyzer.extract_features(time_series)
                        features.update({
                            'neuron_id': neuron_id,
                            'behavior': behavior,
                            'segment_id': segment_idx + 1,
                            'start_time': start_time,
                            'end_time': end_time,
                            'duration': end_time - start_time,
                            'n_samples': len(time_series)
                        })
                        feature_data.append(features)
                    except Exception as e:
                        print(f"    Error processing neuron {neuron_id}: {str(e)}")
                        plt.close('all')  # Ensure figures are closed on error
                        continue
                
                # Create and save features DataFrame for this segment
                if feature_data:
                    features_df = pd.DataFrame(feature_data)
                    features_df.to_excel(os.path.join(segment_dir, 'frequency_features.xlsx'), 
                                      index=False)
                    
                    # Save summary statistics for this segment
                    summary_stats = features_df.describe()
                    summary_stats.to_excel(os.path.join(segment_dir, 'summary_statistics.xlsx'))
            except Exception as e:
                print(f"  Error processing segment {segment_idx+1}: {str(e)}")
                plt.close('all')  # Ensure figures are closed on error
                continue
            finally:
                plt.close('all')  # Always close all figures after processing a segment
        
        # Create behavior summary
        try:
            # Combine all segment features for this behavior
            all_features = []
            for segment_idx, _ in enumerate(segments):
                segment_dir = os.path.join(behavior_dir, f'segment_{segment_idx+1:03d}')
                feature_file = os.path.join(segment_dir, 'frequency_features.xlsx')
                if os.path.exists(feature_file):
                    segment_features = pd.read_excel(feature_file)
                    all_features.append(segment_features)
            
            if all_features:
                # Combine all features and save
                combined_features = pd.concat(all_features, ignore_index=True)
                combined_features.to_excel(os.path.join(behavior_dir, 'all_segments_features.xlsx'),
                                        index=False)
                
                # Calculate and save behavior-level statistics
                behavior_stats = combined_features.groupby('neuron_id').agg({
                    'main_frequency': ['mean', 'std'],
                    'peak_power': ['mean', 'std'],
                    'main_period': ['mean', 'std'],
                    'acf_decay_time': ['mean', 'std'],
                    'n_periods': ['mean', 'std']
                }).round(3)
                
                behavior_stats.to_excel(os.path.join(behavior_dir, 'behavior_statistics.xlsx'))
        except Exception as e:
            print(f"Error creating behavior summary for {behavior}: {str(e)}")
            plt.close('all')
    
    print("\nAnalysis complete!")
    plt.close('all')  # Final cleanup

if __name__ == "__main__":
    main()