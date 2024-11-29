import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.signal import find_peaks, butter, filtfilt
# import SingnalGeneration as sg


maxInterval = 0.2 # s
freq = 0.5 # hz
f_0 = 25
f_1 = 4_200

def genSineWave(freq, samplingRate, time, noiseLevel = 0):
    t = np.linspace(0, time, int(samplingRate * time), endpoint=False)
    signal = np.sin(2 * np.pi * freq * t)
    noise = noiseLevel * np.random.normal(size=t.shape)
    return t, signal + noise

def generateFreqChangeWave(fStart, fEnd, samplingRate, time):
    t = np.linspace(0, time, int(samplingRate * time), endpoint=False)
    freqs = np.linspace(fStart, fEnd, t.size)
    signal = np.sin(2 * np.pi * freqs * t)
    return t, signal

def frequencyDetection(signal, samplingRate):
    N = len(signal)
    dft = fft(signal)
    magnitudes = np.abs(dft[:N // 2])
    freqBins = np.fft.fftfreq(N, 1 / samplingRate)[:N // 2]
    dominantFreq = freqBins[np.argmax(magnitudes)]
    return dominantFreq, magnitudes, freqBins

def frequencyDetectionZeroPadding(signal, samplingRate, padFactor=2):
    N = len(signal)
    paddedSignal = np.pad(signal, (0, N * (padFactor - 1)), 'constant')
    dft = fft(paddedSignal)
    magnitudes = np.abs(dft[:N * padFactor // 2])
    freqBins = np.fft.fftfreq(len(paddedSignal), 1 / samplingRate)[:N * padFactor // 2]
    dominantFreq = freqBins[np.argmax(magnitudes)]
    return dominantFreq, magnitudes, freqBins

def bandpassFilter(signal, samplingRate, lowcut=25, highcut=4200, order=5):
    nyquist = 0.5 * samplingRate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def frequencyDetectionWithFilter(signal, samplingRate):
    filteredSignal = bandpassFilter(signal, samplingRate)
    return frequencyDetection(filteredSignal, samplingRate)


def harmonicAnalysis(signal, samplingRate):
    dominantFreq, magnitudes, freqBins = frequencyDetection(signal, samplingRate)
    harmonics = [dominantFreq * i for i in range(1, 6)]
    mainFundamental = min(harmonics, key=lambda h: abs(h - dominantFreq))
    return mainFundamental, magnitudes, freqBins


def plotResults(t, signal, samplingRate, detectionFunc, title="Frequency Detection"):
    freq, magnitudes, freqBins = detectionFunc(signal, samplingRate)
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, signal)
    plt.title("Time Domain Signal")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    
    plt.subplot(2, 1, 2)
    plt.plot(freqBins, magnitudes)
    plt.title(f"{title}: Dominant Frequency = {freq:.2f} Hz")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.show()

# Test the baseline and modifications
samplingRate = 10_000
time = 1.0
t, signal = genSineWave(freq=440, samplingRate=samplingRate, time=time, noiseLevel=0.1)


# Plot all the data in one window
plt.figure(figsize=(12, 12))

# Baseline Detection
plt.subplot(4, 2, 1)
plt.plot(t, signal)
plt.title("Time Domain Signal - Baseline Detection")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")

plt.subplot(4, 2, 2)
freq, magnitudes, freqBins = frequencyDetection(signal, samplingRate)
plt.plot(freqBins, magnitudes)
plt.title(f"Baseline Detection: Dominant Frequency = {freq:.2f} Hz")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")

# Zero-Padding Detection
plt.subplot(4, 2, 3)
plt.plot(t, signal)
plt.title("Time Domain Signal - Zero-Padding Detection")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")

plt.subplot(4, 2, 4)
freq, magnitudes, freqBins = frequencyDetectionZeroPadding(signal, samplingRate)
plt.plot(freqBins, magnitudes)
plt.title(f"Zero-Padding Detection: Dominant Frequency = {freq:.2f} Hz")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")

# Filtered Detection
plt.subplot(4, 2, 5)
plt.plot(t, signal)
plt.title("Time Domain Signal - Filtered Detection")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")

plt.subplot(4, 2, 6)
freq, magnitudes, freqBins = frequencyDetectionWithFilter(signal, samplingRate)
plt.plot(freqBins, magnitudes)
plt.title(f"Filtered Detection: Dominant Frequency = {freq:.2f} Hz")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")

# Harmonic Analysis Detection
plt.subplot(4, 2, 7)
plt.plot(t, signal)
plt.title("Time Domain Signal - Harmonic Analysis Detection")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")

plt.subplot(4, 2, 8)
freq, magnitudes, freqBins = harmonicAnalysis(signal, samplingRate)
plt.plot(freqBins, magnitudes)
plt.title(f"Harmonic Analysis Detection: Main Fundamental = {freq:.2f} Hz")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")

plt.tight_layout()
plt.show()