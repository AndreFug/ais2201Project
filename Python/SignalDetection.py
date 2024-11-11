import numpy as np
import matplotlib.pyplot as plt
import scipy.signal 
from scipy.fftpack import fft
import librosa
import SingnalGeneration as sg



def detectDominantFrequency(signal, fs, method):
    """
    Args:
        - signal (np.ndarray): Input time-domain signal.
        - fs (float): Sampling frequency in Hz.
        1 = Basic method (baseline)
        2 = Zero-padding method for higher resolution
        3 = Windowing method to reduce spectral leakage
        4 = Interpolation for frequency bin refinement
        5 = Peak detection for robust detection in noisy environments

    Returns:
        - dominant_freq (float): Detected dominant frequency in Hz.
    """
    N = len(signal)
    X = fft(signal)
    magnitudes = np.abs(X[:N // 2])  # Take positive frequencies only

    if method == 1:
        # Basic method (baseline)
        m = np.argmax(magnitudes)
        dominant_freq = m * fs / N

    elif method == 2:
        # Zero-padding method for higher resolution
        zero_padded_N = N * 4  # Padding factor (can be adjusted)
        X_padded = np.fft.fft(signal, zero_padded_N)
        magnitudes_padded = np.abs(X_padded[:zero_padded_N // 2])
        m = np.argmax(magnitudes_padded)
        dominant_freq = m * fs / zero_padded_N

    elif method == 3:
        # Windowing method to reduce spectral leakage
        window = np.blackman(N)  # Using Blackman window
        signal_windowed = signal * window
        X_windowed = fft(signal_windowed)
        magnitudes_windowed = np.abs(X_windowed[:N // 2])
        m = np.argmax(magnitudes_windowed)
        dominant_freq = m * fs / N

    elif method == 4:
        # Interpolation for frequency bin refinement
        m = np.argmax(magnitudes)
        if 1 <= m < (N // 2) - 1:
            alpha = magnitudes[m - 1]
            beta = magnitudes[m]
            gamma = magnitudes[m + 1]
            p = 0.5 * ((alpha - gamma) / (alpha - 2 * beta + gamma))
            dominant_freq = (m + p) * fs / N
        else:
            dominant_freq = m * fs / N

    elif method == 5:
        # Peak detection for robust detection in noisy environments
        peaks, _ = scipy.signal.find_peaks(magnitudes, height=0.1 * max(magnitudes))  # Adjustable threshold
        if peaks.size > 0:
            dominant_peak = peaks[np.argmax(magnitudes[peaks])]
            dominant_freq = dominant_peak * fs / N
        else:
            dominant_freq = 0  # Fallback if no peak is found

    else:
        raise ValueError("Invalid method. Choose from 'basic', 'zero_padding', 'windowing', 'interpolation', or 'peak_detection'.")

    return dominant_freq





# Goes thru all signal types and generates a signal
# signalType = 1
# method = 1

samplingRate = 44100  
duration = 2  
frequency = 560  
noiseLevel = 0.2 

for signalType in range(1, 5):
    for method in range(1, 6):
        t, signal = sg.generateSignal(signalType, frequency, samplingRate, noiseLevel, duration)
        detectedFreq = detectDominantFrequency(signal, samplingRate, method)
        print(f"Detected frequency for signal type {signalType} & method {method}: {detectedFreq:.2f} Hz")
        if method >= 4:
            print("\n\nNew method ")

# # Visualize the generated signal
# plt.figure(figsize=(14, 6))
# plt.plot(t, signal)
# plt.title(f'Signal Type {signalType} (Detected Frequency: {detectedFreq:.2f} Hz)')
# plt.xlabel('Time [s]')
# plt.ylabel('Amplitude')
# plt.show()