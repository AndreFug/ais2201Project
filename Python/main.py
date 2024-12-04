# Imports
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import butter, lfilter, firwin
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from signalGenerator import *
from baseline_algorithm import *

audioFiles = {
    1: 'A4',
    2: 'A5',
    3: 'B',
    4: 'C#',
    5: 'D',
    6: 'E',
    7: 'F#',
    8: 'G#',
    9: 'Zauberflöte_vocal'
}

def openAudio(filename, audioType):
    """Open audio file based on audio type and filename."""
    if audioType <= 8:
        fs, sampleData = wavfile.read(f'./Python/sample_audio/{filename}_oboe.wav')
    else:
        fs, sampleData = wavfile.read('./Python/sample_audio/Zauberflöte_vocal.wav')
    N = len(sampleData)
    return fs, sampleData, N

def computeDFT(sampleData, N, fs):
    """Compute the Discrete Fourier Transform (DFT) of the signal."""
    dft = np.fft.fft(sampleData[:N])[:N//2]
    freqs = np.fft.fftfreq(N, d=1/fs)[:N//2]
    mags = np.abs(dft)
    return freqs, mags

def detectDominantFrequency(sampleData, fs, N):
    """Detect the dominant frequency in the signal."""
    X = np.fft.fft(sampleData[:N])[:N//2]
    mags = np.abs(X)
    idx = np.argmax(mags)  # Index of the peak frequency
    dominantFreq = idx * fs / N
    return dominantFreq

class Filter:
    """
    Class to implement filtering functions.
    """
    @staticmethod
    def zeroPad(sampleData, targetLength):
        filterType = "Zero padding"
        filteredData = np.pad(sampleData, (0, targetLength - len(sampleData)), 'constant')
        return filteredData, filterType 
    
    @staticmethod
    def bandpassFilter(sampleData, fs, lowcut=25, highcut=4200):    # Given parameters from task
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(4, [low, high], btype='band')
        filterType = "Bandpass"
        filteredData = lfilter(b, a, sampleData)
        return filteredData, filterType
    
    @staticmethod
    def lowHighPassFilter(sampleData, fs, btype='low', cutoff=0):
        """
        Apply a low- or high-pass Butterworth filter.
        """
        if btype == 'low':
            cutoff = 25
            filterType = "Low-pass"
        else:  
            cutoff = 4200
            filterType = "High-pass"
        nyquist = fs * 0.5
        normalCutoff = cutoff / nyquist
        b, a = butter(4, normalCutoff, btype)
        filteredData = lfilter(b, a, sampleData)
        return  filteredData, filterType
    
    @staticmethod
    def firFilter(sampleData, fs, cutoff, numtaps, btype):
        """
        Apply a Finite Impulse Response (FIR) filter.
        """
        nyquist = 0.5 * fs
        normalCutoff = cutoff / nyquist
        filterType = "FIR"
        if btype == 'low':
            taps = firwin(numtaps, normalCutoff)
        elif btype == 'high':
            taps = firwin(numtaps, normalCutoff, pass_zero=False)
        else:
            raise ValueError("btype must be 'low' or 'high'")
        filteredData = lfilter(taps, 1.0, sampleData)
        return filteredData, filterType

def plotSpectrum(title, freqs, mags, freqsFiltered, magsFiltered, dominantFreqBefore, dominantFreqAfter):
    """Plot the frequency spectrum with dominant frequencies marked."""
    plt.figure()
    plt.plot(freqs, mags, label='Original')
    plt.plot(freqsFiltered, magsFiltered, color='red', linestyle='dashed', label='Filtered')
    plt.axvline(x=dominantFreqBefore, color='blue', linestyle=':', label=f'Dominant Freq Before: {dominantFreqBefore:.2f} Hz')
    plt.axvline(x=dominantFreqAfter, color='red', linestyle=':', label=f'Dominant Freq After: {dominantFreqAfter:.2f} Hz')
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    plt.show()

# Plot all the audio files and their frequency spectrum before and after filtering
def plotAll(audioFiles):
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for i in range(1, 10):
        audioType = i
        filename = audioFiles.get(audioType, "")
        fs, sampleData, N = openAudio(filename, audioType)
        freqs, mags = computeDFT(sampleData, N, fs)
        
        # Apply filter (you can change the filter type and parameters here)
        # For demonstration, we'll apply a bandpass filter that includes the expected dominant frequency
        filteredData, filterType = Filter.bandpassFilter(sampleData, fs, lowcut=100, highcut=5000)
        freqsFiltered, magsFiltered = computeDFT(filteredData, N, fs)
        
        # Detect dominant frequencies before and after filtering
        dominantFreqBefore = detectDominantFrequency(sampleData, fs, N)
        dominantFreqAfter = detectDominantFrequency(filteredData, fs, N)

        # Plotting
        row = (i - 1) // 3
        col = (i - 1) % 3
        ax = axes[row, col]
        ax.plot(freqs, mags, color='navy', label='Original')
        ax.plot(freqsFiltered, magsFiltered, color='darkred', linestyle='dashed', label='Filtered')
        ax.axvline(x=dominantFreqBefore, color='magenta', linestyle='-', label=f'Freq Before: {dominantFreqBefore:.2f} Hz')
        ax.axvline(x=dominantFreqAfter, color='gold', linestyle=':', label=f'Freq After: {dominantFreqAfter:.2f} Hz')
        ax.set_title(f'{filename} audio')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Amplitude')
        ax.set_xlim(0, 5000)    # Limit to 5 kHz    
        ax.grid(True)
        ax.legend()

    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    fig.suptitle(f'Filter type: {filterType}', fontsize=20)
    plt.show()

# Generate signals
duration = 1
fs = 44100
freq = 440  # Hz
noise = 0.5

t_pureSine, gen_pureSine, signalType_pureSine = generateSignal('Pure sine', freq, fs, noise, duration)
t_abruptSine, gen_abruptSine, signalType_abrupt = generateSignal('Abrupt sine', freq, fs, noise, duration)
t_music, gen_music, signalType_music = generateSignal('Music', freq, fs, noise, duration)
t_vocal, gen_vocal, signalType_vocal = generateSignal('Vocal', freq, fs, noise, duration)

generatedSignals = {
    1: (t_pureSine, gen_pureSine, signalType_pureSine),
    2: (t_abruptSine, gen_abruptSine, signalType_abrupt),
    3: (t_music, gen_music, signalType_music),
    4: (t_vocal, gen_vocal, signalType_vocal)
}

def plotGenSignals(generatedSignals):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    for i, (key, (t, signal, signalType)) in enumerate(generatedSignals.items()):
        N = len(signal)
        delta_t = t[1] - t[0]
        fs = 1 / delta_t
        freqs, mags = computeDFT(signal, N, fs)
        
        # Apply filter (you can change the filter type and parameters here)
        # For demonstration, we'll use a low-pass filter
        filteredData, filterType = Filter.lowHighPassFilter(signal, fs, btype='low', cutoff=1000)
        freqsFiltered, magsFiltered = computeDFT(filteredData, N, fs)
        
        # Detect dominant frequencies before and after filtering
        dominantFreqBefore = detectDominantFrequency(signal, fs, N)
        dominantFreqAfter = detectDominantFrequency(filteredData, fs, N)
        
        # Plotting
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        ax.plot(freqs, mags, color='blue', label='Original FFT')
        ax.plot(freqsFiltered, magsFiltered, color='red', linestyle='dashed', label='Filtered DFT')
        ax.axvline(x=dominantFreqBefore, color='magenta', linestyle='-', label=f'Freq Before: {dominantFreqBefore:.2f} Hz')
        ax.axvline(x=dominantFreqAfter, color='gold', linestyle=':', label=f'Freq After: {dominantFreqAfter:.2f} Hz')
        ax.set_title(signalType)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Amplitude')
        ax.set_xlim(0, 5000)  # Limit to 5 kHz
        ax.grid(True)
        ax.legend()

    plt.subplots_adjust(hspace=0.69, wspace=0.420)
    fig.suptitle(f'Filter type: {filterType}', fontsize=20)
    plt.show()

plotAll(audioFiles)
# plotGenSignals(generatedSignals)
