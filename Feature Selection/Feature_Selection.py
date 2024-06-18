import numpy as np
import pandas as pd
from scipy.signal import find_peaks, welch
from scipy.stats import skew, kurtosis

def calculate_features(segment):
    N = len(segment)
    mu = np.mean(segment)
    sigma = np.std(segment)
    
    features = {
        'min': np.min(segment),
        'max': np.max(segment),
        'mean': mu,
        'variance': np.var(segment, ddof=0),
        'skewness': np.sum((segment - mu)**3) / (N * sigma**3),
        'kurtosis': np.sum((segment - mu)**4) / (N * sigma**4) - 3
    }
    
    # Autocorrelation for lags 0 to 10
    autocorrs = np.correlate(segment - mu, segment - mu, mode='full')[N-1:N+10] / (N - np.arange(0, 11))
    features.update({f'autocorr_{delta}': autocorrs[delta] for delta in range(11)})
    
    # DFT calculation
    dft = np.fft.fft(segment)
    spectrum = np.abs(dft)
    frequencies = np.fft.fftfreq(N, d=1/25)  # Assuming sample rate is 25 Hz
    peaks, _ = find_peaks(spectrum)
    top_peaks = peaks[:5] if len(peaks) > 5 else peaks
    
    features.update({f'peak_{i}': spectrum[top_peak] for i, top_peak in enumerate(top_peaks)})
    features.update({f'freq_{i}': frequencies[top_peak] for i, top_peak in enumerate(top_peaks)})
    
    return features

def preprocess_and_extract_features(file_path):
    df = pd.read_csv(file_path)
    df['A(T)'] = np.sqrt(df['Acc_X']**2 + df['Acc_Y']**2 + df['Acc_Z']**2)
    peak_index = df['A(T)'].idxmax()
    start = max(0, peak_index - 50)
    end = min(len(df), peak_index + 51)
    window_df = df.loc[start:end].reset_index(drop=True)
    
    extracted_features = {column: calculate_features(window_df[column]) for column in window_df.columns if column not in ['Counter', 'A(T)']}
    
    # Convert dictionary to DataFrame
    features_df = pd.DataFrame.from_dict(extracted_features, orient='index')
    output_path = f"extracted_features_{file_path.split('/')[-1]}"
    features_df.to_csv(output_path)
    
    return output_path

# Example usage
file_paths = ["path_to_your_dataset1.csv", "path_to_your_dataset2.csv", ...]
all_features = {path: preprocess_and_extract_features(path) for path in file_paths}
