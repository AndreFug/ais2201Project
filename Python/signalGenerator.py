import numpy as np
import librosa, librosa.display


def generateSignal(signalType, frequency, samplingRate, noiseLevel, duration):
    
    '''
    Args:
        Pure sine wave with varying levels of noise
        Sine wave with abrupt change in frequency
        Musical instrument with varying levels of noise
        Vocal
        - frequency (int): Frequency of the sine wave in Hz.
        - samplingRate (int): Sampling rate in Hz.
        - noiseLevel (float): Standard deviation of the noise.
        - duration (int): Duration of the signal in seconds.
    Returns:
        - t (np.ndarray): Time vector.
        - signalOut (np.ndarray): Generated signal.
    '''
    t = np.linspace(0, duration, int(samplingRate * duration), endpoint=False)
    if signalType == 'Pure sine':
        # Pure sine wave with varying levels of noise
        sineWave = np.sin(2 * np.pi * frequency * t)
        noise = np.random.normal(0, noiseLevel, sineWave.shape)
        signalOut = sineWave + noise
    
    elif signalType == 'Abrupt sine':
        # Sine wave with abrupt change in frequency
        midpoint = len(t) // 2
        sineWave1 = np.sin(2 * np.pi * frequency * t[:midpoint])
        sineWave2 = np.sin(2 * np.pi * (frequency * 2) * t[midpoint:])
        signalOut = np.concatenate((sineWave1, sineWave2))
    
    elif signalType == 'Music':
        # Musical instrument with varying levels of noise
        y, _ = librosa.load(librosa.example('trumpet'), sr=samplingRate, duration=duration)
        noise = np.random.normal(0, noiseLevel, y.shape)
        signalOut = y + noise
    
    elif signalType == 'Vocal':
        # Vocal
        y, _ = librosa.load(librosa.example('vibeace'), sr=samplingRate, duration=duration)
        signalOut = y
    else:
        raise ValueError("Write the correct signal type, moron :)")
    return t, signalOut
