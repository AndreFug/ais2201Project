from baseline_algorithm import *
import numpy as np
from scipy.signal import lfilter, firwin

def zero_padding(signal, fs, N):
    """
    Zero pads the signal to have a length of N samples.
    
    Parameters:
    - signal: ndarray, the input signal.
    - fs: float, the sampling rate of the signal in Hz.
    - N: int, the desired length of the signal in samples.
    
    Returns:
    - padded_signal: ndarray, the zero-padded signal.
    """
    n = len(signal)
    if n >= N:
        return signal
    else:
        padded_signal = np.zeros(N)
        padded_signal[:n] = signal
        return padded_signal
    


def hanning(signal, fs, N):
    """
    Apply the Hanning window to a signal.
    
    Parameters:
    - signal: ndarray, the input signal.
    - fs: float, the sampling rate of the signal in Hz.
    
    Returns:
    - windowed_signal: ndarray, the windowed signal.
    """
    n = len(signal)
    window = np.hanning(n)
    windowed_signal = signal * window
    return windowed_signal

def fir_filter(signal, fs, cutoff, numtaps=51, filter_type='bandpass'):
    """
    Apply an FIR filter to a signal.

    Parameters:
        signal (numpy.ndarray): The input time-domain signal.
        fs (float): The sampling frequency in Hz.
        cutoff (float or tuple): The cutoff frequency/frequencies in Hz. For band-pass or band-stop, provide a tuple (low, high).
        numtaps (int): The number of filter coefficients (order + 1).
        filter_type (str): The type of filter ('lowpass', 'highpass', 'bandpass', 'bandstop').

    Returns:
        numpy.ndarray: The filtered signal.
    """
    if isinstance(cutoff, tuple):
        cutoff = [f / (fs / 2) for f in cutoff] 
    else:
        cutoff = cutoff / (fs / 2)

    fir_coeff = firwin(numtaps, cutoff, pass_zero=(filter_type == 'lowpass' or filter_type == 'bandstop'))

    filtered_signal = lfilter(fir_coeff, 1.0, signal)

    return filtered_signal, fir_coeff


