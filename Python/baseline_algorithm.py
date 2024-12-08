import numpy as np
from filters import *

def freq_detection(x_n: np.ndarray, fs: int, N: int = 1024)->float:
    """
    Identifies the primary sinusoidal component in a signal x[n]
    over time by calculting successive N-point DFTs of x[n], and
    selecting the frequency component with the highest magnitude. 

    Parameters:
    x_n - signal samples x[n] to be analyzed
    fs - sampling frequency
    N - DFT window size in number of samples 
        Defaults to 1024 samples

    Returns:
    timestamps - ndarray of floats
        Points in time at which frequency contents were estimated.
    freqs - ndarray of floats
        Most prominent frequency detected for corresponding timestamp
        values.
    """
    timestamps = []
    freqs = []
    for window_start in range(0, len(x_n), N):
        window_end = window_start + N if len(x_n) >= N else len(x_n)
        x_slice = x_n[window_start:window_end]
        X_m = np.fft.rfft(x_slice, n = N)  
        X_m[0] = 0 
        m_peak = np.argmax(np.abs(X_m))  
        freqs.append(m_peak/N*fs)  
        timestamps.append(window_end/fs)
    return np.array(timestamps), np.array(freqs)

def freq_detection_zero_pad(x_n: np.ndarray, fs: int, N: int = 1024):
    """
    Identifies the primary sinusoidal component in a signal x[n] using zero padding.
    """
    timestamps = []
    freqs = []
    for window_start in range(0, len(x_n), N):
        window_end = min(window_start + N, len(x_n))
        x_slice = x_n[window_start:window_end]

        x_slice = zero_padding(x_slice, len(x_n), N)

        X_m = np.fft.rfft(x_slice, n=N)
        X_m[0] = 0 

        m_peak = np.argmax(np.abs(X_m))
        freqs.append(m_peak / N * fs)
        timestamps.append(window_end / fs)

    return np.array(timestamps), np.array(freqs)

def freq_detection_hanning(x_n: np.ndarray, fs: int, N: int = 1024):
    """
    Identifies the primary sinusoidal component in a signal x[n] using a Hanning window.
    """
    timestamps = []
    freqs = []
    for window_start in range(0, len(x_n), N):
        window_end = min(window_start + N, len(x_n))
        x_slice = x_n[window_start:window_end]

        x_slice = hanning(x_n, fs, N)

        X_m = np.fft.rfft(x_slice, n=N)
        X_m[0] = 0  

        m_peak = np.argmax(np.abs(X_m))
        freqs.append(m_peak / N * fs)
        timestamps.append(window_end / fs)

    return np.array(timestamps), np.array(freqs)


def freq_detection_fir_filter(x_n: np.ndarray, fs: int, h: np.ndarray, N: int = 1024):
    """
    Identifies the primary sinusoidal component in a signal x[n] using an FIR filter.
    """
    timestamps = []
    freqs = []

    # Extract filtered signal
    x_filtered, _ = fir_filter(x_n, fs, (25, 4200))  # Adjust here

    for window_start in range(0, len(x_filtered), N):
        window_end = min(window_start + N, len(x_filtered))
        x_slice = x_filtered[window_start:window_end]

        # Check the shape and content of x_slice for debugging
        if not isinstance(x_slice, np.ndarray):
            raise ValueError(f"x_slice must be a numpy array, got {type(x_slice)}")
        if len(x_slice) == 0:
            continue  # Skip empty slices

        # Perform FFT
        X_m = np.fft.rfft(x_slice, n=N)
        X_m[0] = 0  # Remove DC component

        # Find peak frequency
        m_peak = np.argmax(np.abs(X_m))
        freqs.append(m_peak / N * fs)
        timestamps.append(window_end / fs)

    return np.array(timestamps), np.array(freqs)




