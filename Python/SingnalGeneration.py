import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf

def generateSignal(signalType, frequency, samplingRate, noiseLevel, duration):
    t = np.linspace(0, duration, int(samplingRate * duration), endpoint=False)

    if signalType == 1:
        # Pure sine wave with varying levels of noise
        sine_wave = np.sin(2 * np.pi * frequency * t)
        noise = np.random.normal(0, noiseLevel, sine_wave.shape)
        signalOut = sine_wave + noise
    
    elif signalType == 2:
        # Sine wave with abrupt change in frequency
        midpoint = len(t) // 2
        sine_wave_1 = np.sin(2 * np.pi * frequency * t[:midpoint])
        sine_wave_2 = np.sin(2 * np.pi * (frequency * 2) * t[midpoint:])
        signalOut = np.concatenate((sine_wave_1, sine_wave_2))
    
    elif signalType == 3:
        # Musical instrument with varying levels of noise
        y, _ = librosa.load(librosa.example('trumpet'), sr=samplingRate, duration=duration)
        noise = np.random.normal(0, noiseLevel, y.shape)
        signalOut = y + noise
    
    elif signalType == 4:
        # Vocal
        y, _ = librosa.load(librosa.example('vibeace'), sr=samplingRate, duration=duration)
        signalOut = y

    else:
        raise ValueError("Invalid signal type. Please choose a value between 1 and 4.")

    return t, signalOut

# noisySine_t, noisySine_signalOut = generateSignal(1, 440, 44100, 0.2, 2)
