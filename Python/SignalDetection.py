import numpy as np
import matplotlib.pyplot as plt
import scipy.signal 
from scipy.fftpack import fft




def detectDominantFrequency(signal, fs, method):
    """
    Args:
        - signal (np.ndarray): Input time-domain signal.
        - fs (float): Sampling frequency in Hz.
        - method (int):
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
        dominantFreq = m * fs / N

    elif method == 2:
        # Zero-padding method for higher resolution
        zeroPaddedN = N * 4  # Padding factor (can be adjusted)
        XPadded = np.fft.fft(signal, zeroPaddedN)
        magnitudesPadded = np.abs(XPadded[:zeroPaddedN // 2])
        m = np.argmax(magnitudesPadded)
        dominantFreq = m * fs / zeroPaddedN

    elif method == 3:
        # Windowing method to reduce spectral leakage
        window = np.blackman(N)  # Using Blackman window
        signalWindowed = signal * window
        X_windowed = fft(signalWindowed)
        magnitudesWindowed = np.abs(X_windowed[:N // 2])
        m = np.argmax(magnitudesWindowed)
        dominantFreq = m * fs / N

    elif method == 4:
        # Interpolation for frequency bin refinement
        m = np.argmax(magnitudes)
        if 1 <= m < (N // 2) - 1:
            alpha = magnitudes[m - 1]
            beta = magnitudes[m]
            gamma = magnitudes[m + 1]
            p = 0.5 * ((alpha - gamma) / (alpha - 2 * beta + gamma))
            dominantFreq = (m + p) * fs / N
        else:
            dominantFreq = m * fs / N

    elif method == 5:
        # Peak detection for robust detection in noisy environments
        peaks, _ = scipy.signal.find_peaks(magnitudes, height=0.1 * max(magnitudes))  # Adjustable threshold
        if peaks.size > 0:
            dominantPeak = peaks[np.argmax(magnitudes[peaks])]
            dominantFreq = dominantPeak * fs / N
        else:
            dominantFreq = 0  # Fallback if no peak is found

    else:
        raise ValueError("Invalid method. Choose from 'basic', 'zero_padding', 'windowing', 'interpolation', or 'peak_detection'.")

    return dominantFreq





# Goes thru all signal types and generates a signal
# signalType = 1
# method = 1

samplingRate = 44100  
duration = 2  
frequency = 560  
noiseLevel = 0.2 



# # Visualize the generated signal
# plt.figure(figsize=(14, 6))
# plt.plot(t, signal)
# plt.title(f'Signal Type {signalType} (Detected Frequency: {detectedFreq:.2f} Hz)')
# plt.xlabel('Time [s]')
# plt.ylabel('Amplitude')
# plt.show()