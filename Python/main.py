import scipy.io.wavfile as wavfile
from scipy.signal import butter, lfilter, firwin, savgol_filter
from IPython.display import Audio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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
    def bandpassFilter(sampleData, fs, lowcut, highcut):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(4, [low, high], btype='band')
        filterType = "Bandpass"
        filteredData = lfilter(b, a, sampleData)
        return filteredData, filterType
    

    @staticmethod
    def low_highPassFilter(sampleData, fs, cutoff, btype):
        """
        Apply a low- or high-pass Butterworth filter.
        """
        nyquist = fs * 0.5
        normal_cutoff = cutoff / nyquist
        b, a = butter(4, normal_cutoff, btype)

        filteredData = lfilter(b, a, sampleData)
        if btype == 'low':
            filterType = "Low-pass"
        else:  
            filterType = "High-pass"
        return  filteredData, filterType

def test_frequency_detection(audioFiles):
    results = {}
    for audioType, filename in audioFiles.items():
        fs, sampleData, N, nyquist = openAudio(filename, audioType)
        freqs, mags = computeDFT(sampleData, N, fs)
        fundamental = freqs[np.argmax(mags)]
        results[filename] = fundamental
        print(f'{filename}: Detected frequency = {fundamental} Hz')
    return results

def plot_spectrum(title, freqs, mags, freqsFiltered, magsFiltered):
    """Plot the frequency spectrum."""
    plt.figure()
    plt.plot(freqs, mags)
    plt.plot(freqsFiltered, magsFiltered, color='red', linestyle='dashed')
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend(['Original', 'Filtered'])
    plt.show()

# Plots all the audio files and their frequency spectrum before and after filtering
def plotAll(audioFiles):
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for i in range(1, 10):
        audioType = i
        filename = audioFiles.get(audioType, "")
        fs, sampleData, N = openAudio(filename, audioType)
        freqs, mags = computeDFT(sampleData, N, fs)
        filteredData, filteType = Filter.low_highPassFilter(sampleData, fs, 1000, 'high')
        freqsFiltered, magsFiltered = computeDFT(filteredData, N, fs)
        row = (i - 1) // 3
        col = (i - 1) % 3
        ax = axes[row, col]
        ax.plot(freqs, mags, color='blue')
        ax.plot(freqsFiltered, magsFiltered, color='red', linestyle=(0, (1, 5)))
        ax.set_title(f'{filename} audio')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Amplitude')
        ax.set_xlim(0, 5000)    # Nothing fun after 5kHz
        ax.grid(True)
            
    plt.subplots_adjust(hspace=0.69, wspace=0.420)  # hehe funny numbers
    fig.suptitle(f'Filter type: {filteType}', fontsize=20)
    line_original = Line2D([0], [0], color='blue', lw=2)
    line_filtered = Line2D([0], [0], color='red', lw=2, linestyle=(0, (1,5)))
    fig.legend([line_original, line_filtered], ['Original', 'Filtered'], loc='upper right', fontsize=14, frameon=True, ncol=1)
    plt.show()

plotAll(audioFiles)
