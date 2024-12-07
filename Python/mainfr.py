from baseline_algorithm import freq_detection
import numpy as np
import matplotlib.pyplot as plt
from signalGenerator import generateSignal
from scipy.io import wavfile
from scipy.signal import resample_poly


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

SNRs = []
avgEst = []
avgError = []
vars = []

t, signal, type = generateSignal('Pure sine', 440, 44100, 0.1, 1)



fileName = audioFiles[9]
if fileName == 'Zauberflöte_vocal':
    fsOriginal, audioSignal = wavfile.read(f'./Python/sample_audio/Zauberflöte_vocal.wav')
else:
    fsOriginal, audioSignal = wavfile.read(f'./Python/sample_audio/{fileName}_oboe.wav')

audioSignal = audioSignal.astype(np.float32)
audioSignal /= np.max(np.abs(audioSignal))

if audioSignal.ndim > 1:
    audioSignal = audioSignal[:, 0]

fs_target = 16000
if fsOriginal != fs_target:
    audioSignalResampled = resample_poly(audioSignal, fs_target, fsOriginal)
else:
    audioSignalResampled = audioSignal

duration = 2.2 
N_samples = int(duration * fs_target)
if len(audioSignalResampled) >= N_samples:
    audioSignalResampled = audioSignalResampled[:N_samples]
else:
    audioSignalResampled = np.pad(audioSignalResampled, (0, N_samples - len(audioSignalResampled)), 'constant')

signalPwr = np.mean(audioSignalResampled ** 2)

noisePwr = np.logspace(-2, 2, 20)

SNRs = []
avgEst = []
avgError = []
vars = []

trueFrequency = 440  # For 'A4_oboe.wav', the fundamental frequency is 440 Hz

N_DFT = len(audioSignal)

# Perform frequency estimation over varying noise levels
for noiseVar in noisePwr:
    # Generate white Gaussian noise
    noise = np.random.normal(scale=np.sqrt(noiseVar), size=audioSignalResampled.shape)
    # Add noise to the signal
    noisySignal = audioSignalResampled + noise
    # Estimate the frequency using the freq_detection function
    timestamps, freqs, mags = freq_detection(noisySignal, fs_target, N=1024)
    # Clip the estimated frequency to the valid range
    freqs = np.clip(freqs, 0, fs_target / 2)
    # Calculate statistics
    avg_est_freq = np.mean(freqs)
    variance = np.var(freqs)
    error = np.abs(avg_est_freq - trueFrequency)
    snr = signalPwr / noiseVar
    # Store results
    SNRs.append(snr)
    avgEst.append(avg_est_freq)
    avgError.append(error)
    vars.append(variance)

# Convert SNRs to inverse SNRs for plotting
inv_SNRs = 1 / np.array(SNRs)

# Plot the results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Plot average estimated frequency with variance shading
ax1.plot(inv_SNRs, avgEst, color='r')
ax1.fill_between(inv_SNRs, np.array(avgEst) - np.sqrt(vars),
                 np.array(avgEst) + np.sqrt(vars), color='red', alpha=0.2)
ax1.set_xscale('log')
ax1.set_xlim([inv_SNRs.min(), inv_SNRs.max()])
ax1.set_ylim([0, fs_target / 2])
# ax1.set_title(f'Average Frequency Estimate for "{audioFiles[fileName]}_oboe.wav" ({duration} sec)')
ax1.set_ylabel('Frequency Estimate (Hz)')
ax1.grid(True)

# Plot average error
ax2.plot(inv_SNRs, avgError, color='b')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlim([inv_SNRs.min(), inv_SNRs.max()])
ax2.set_ylim([1e-1, 1e4])
# ax2.set_title(f'Average Error for "{audioFiles[fileName]}_oboe.wav" ({duration} sec)')
ax2.set_xlabel('1/SNR')
ax2.set_ylabel('Error (Hz)')
ax2.grid(True)

plt.tight_layout()
plt.show()