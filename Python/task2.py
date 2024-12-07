from baseline_algorithm import *
from signalGenerator import *
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa, librosa.display
import scipy.io.wavfile as wavfile
from scipy.signal import butter, lfilter, firwin
from scipy.fft import fft, fftfreq

audioFiles ={
    './Python/sample_audio/A4_oboe.wav',
    './Python/sample_audio/A5_oboe.wav',
    './Python/sample_audio/B_oboe.wav',
    './Python/sample_audio/C#_oboe.wav',
    './Python/sample_audio/D_oboe.wav',
    './Python/sample_audio/E_oboe.wav',
    './Python/sample_audio/F#_oboe.wav',
    './Python/sample_audio/G#_oboe.wav',
    './Python/sample_audio/Zauberflöte_vocal.wav'
}