from baseline_algorithm import *
from functions import *
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly, firwin
import matplotlib.pyplot as plt

fs = 1e4
duration = 2
f1 = 100
f2 = 200
amplitude = 1
noise_variance = 0.33

def test_freq_sine_noisey():

    tf = 440

    signal = amplitude * np.sin(2 * np.pi * tf * np.arange(int(fs * duration)) / fs)
    SNR_values = np.logspace(-2, 2, 10)
    noise_variance = (amplitude**2 / (2 * SNR_values))
    avgEst = []
    avgErrors = []
    sickaBars = []
    var = []
    SNRs = []

    for noise_variance in noise_variance:
        noise = np.random.normal(0, np.sqrt(noise_variance), len(signal))
        xn = signal + noise
        timestamps, freqs = freq_detection(xn, fs)
        avg_freq = np.mean(freqs)
        avgEst.append(avg_freq)

        error = np.abs(avg_freq - tf)
        avgError = np.mean(error)
        avgErrors.append(avgError)

        sickaBars.append(np.std(error))
        
    invSNR = 1 / SNR_values
    plt.subplot(2, 1, 1)
    plt.plot(invSNR, avgEst)
    plt.fill_between(invSNR,
                    np.array(avgEst) - np.array(sickaBars),
                    np.array(avgEst) + np.array(sickaBars), 
                    color='red', alpha=0.2)
    plt.xscale('log') 
    plt.legend
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(invSNR, avgErrors)
    plt.xscale('log')
    plt.legend
    plt.grid(True)

    plt.show()


def test_sine_abrupt():
    N_total = int(fs * duration)
    time = np.arange(N_total) / fs
    N_change = int(0.5 * fs)
    signal = np.zeros(N_total)
    signal[:N_change] = amplitude * np.sin(2 * np.pi * f1 * time[:N_change])
    signal[N_change:] = amplitude * np.sin(2 * np.pi * f2 * time[N_change:])
    noise = np.random.normal(0, noise_variance, N_total)
    xn = signal + noise

    n_values = [1024, 2048]
    tf = np.where(time < 0.5, f1, f2)

    plt.figure(figsize=(10, 5))
    plt.plot(time, tf, label='True Frequency $f(t)$', color='blue')

    for N in n_values:
        timestamps, freqs = freq_detection_zero_pad(xn, fs, N=N)
        plt.plot(timestamps, freqs, label=f'Estimated Frequency (N={N})')

    plt.legend()
    plt.grid(True)
    plt.show()

def test_note():
    audio_file = './Python/sample_audio/A4_oboe.wav'
    fs_original, audio_signal = wavfile.read(audio_file)
    audio_signal = audio_signal.astype(np.float32) / np.max(np.abs(audio_signal))

    if audio_signal.ndim > 1:
        audio_signal = audio_signal[:, 0]

    fs_target = 16_000
    duration = 2.2  
    n_samples = int(duration * fs_target)
    audio_signal_resampled = resample_poly(audio_signal, fs_target, fs_original) if fs_original != fs_target else audio_signal

    audio_signal_resampled = np.pad(audio_signal_resampled[:n_samples],
                                    (0, max(0, n_samples - len(audio_signal_resampled))),
                                    'constant')

    signal_pwr = np.mean(audio_signal_resampled ** 2)

    noise_pwr = np.logspace(-2, 2, 20)

    avg_estimates = []
    avg_errors = []
    variances = []
    snr_values = []

    avg_estimates_hanning = []
    avg_errors_hanning = []
    variances_hanning = []

    true_frequency = 440
    n_dft = 2048
    f0 = estimate_fundamental_frequency(audio_signal_resampled, fs_target)
    print(f0)
    for noise_var in noise_pwr:
        noise = np.random.normal(scale=np.sqrt(noise_var), size=len(audio_signal_resampled))
        noisy_signal = audio_signal_resampled + noise**2
        timestamps, freqs = freq_detection(noisy_signal, fs_target, N=n_dft)
        freqs = np.clip(freqs, 0, fs_target / 2)

        avg_freq = np.mean(freqs)
        variance = np.var(freqs)
        error = np.abs(avg_freq - true_frequency)

        snr = signal_pwr / noise_var
        snr_values.append(snr)

        avg_estimates.append(avg_freq)
        avg_errors.append(error)
        variances.append(variance)

        h = firwin(51, [c / (fs_target / 2) for c in (25, 4200)], pass_zero=False)
        # timestamps, freqs_hanning = freq_detection_fir_filter(noisy_signal, fs, h, N=n_dft)
        # timestamps, freqs_hanning = freq_detection_hanning(noisy_signal, fs, N=n_dft)
        # timestamps, freqs_hanning = freq_detection_zero_pad(noisy_signal, fs, N=n_dft)
        filtered_signal = compined_filter(noisy_signal, fs, n_dft)
        f0_filtered = estimate_fundamental_frequency(filtered_signal, fs)
        print(f0_filtered)
        freqs_hanning = freq_detection(filtered_signal, fs_target, N=n_dft)
        freqs_hanning = np.clip(freqs_hanning, 0, fs_target / 2)

        avg_freq_hanning = np.mean(freqs_hanning)
        variance_hanning = np.var(freqs_hanning)
        error_hanning = np.abs(avg_freq_hanning - true_frequency)

        avg_estimates_hanning.append(avg_freq_hanning)
        avg_errors_hanning.append(error)
        variances_hanning.append(variance)

    inv_snr_values = 1 / np.array(snr_values)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    print(f'Before filtering{freqs}')
    print(f'After filtering{freqs_hanning}')
    ax1.plot(inv_snr_values, avg_estimates, color='r')
    ax1.plot(inv_snr_values, avg_estimates_hanning, color='g', linestyle='--')
    ax1.fill_between(inv_snr_values,
                     np.array(avg_estimates) - np.sqrt(variances),
                     np.array(avg_estimates) + np.sqrt(variances),
                     color='red', alpha=0.2)
    ax1.set_xscale('log')
    ax1.set_xlim([inv_snr_values.min(), inv_snr_values.max()])
    ax1.set_ylim([0, fs_target / 2])
    ax1.set_title('Average frequency estimate for recorded audio')
    ax1.set_ylabel('Frequency Estimate (Hz)')
    ax1.grid(True)
    plt.legend(['Before filtering', 'After filtering'])

    ax2.plot(inv_snr_values, avg_errors, color='b')
    ax2.plot(inv_snr_values, avg_errors_hanning, color='g', linestyle='--')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlim([inv_snr_values.min(), inv_snr_values.max()])
    ax2.set_ylim([1e-1, 1e4])
    ax2.set_title('Average error for recorded audio')
    ax2.set_xlabel('1/SNR')
    ax2.set_ylabel('Error (Hz)')
    ax2.grid(True)

    plt.tight_layout()

    plt.show()


def vocal():
    audio_file = './Python/sample_audio/Zauberflöte_vocal.wav'
    fs_original, audio_signal = wavfile.read(audio_file)
    audio_signal = audio_signal.astype(np.float32) / np.max(np.abs(audio_signal))

    if audio_signal.ndim > 1:
        audio_signal = audio_signal[:, 0]

    fs_target = 16_000
    duration = 2.5 
    n_samples = int(duration * fs_target)
    audio_signal_resampled = resample_poly(audio_signal, fs_target, fs_original) if fs_original != fs_target else audio_signal

    audio_signal_resampled = np.pad(audio_signal_resampled[:n_samples],
                                    (0, max(0, n_samples - len(audio_signal_resampled))),
                                    'constant')
    f0 = estimate_fundamental_frequency(audio_signal_resampled, fs_target)
    print(f0)
    signal_pwr = np.mean(audio_signal_resampled ** 2)

    noise_pwr = np.logspace(-2, 2, 10) * signal_pwr

    avg_estimates = []
    avg_errors = []
    variances = []
    snr_values = []

    true_frequency = 0  
    n_dft = 2048  

    for noise_var in noise_pwr:
        noise = np.random.normal(scale=np.sqrt(noise_var), size=len(audio_signal_resampled))
        noisy_signal = audio_signal_resampled + noise
        timestamps, freqs = freq_detection_hanning(noisy_signal, fs_target, N=n_dft)
        freqs = np.clip(freqs, 0, fs_target / 2)

        avg_freq = np.mean(freqs)
        variance = np.var(freqs)
        error = np.abs(avg_freq - true_frequency)

        snr = signal_pwr / noise_var
        snr_values.append(snr)

        avg_estimates.append(avg_freq)
        avg_errors.append(error)
        variances.append(variance)

    inv_snr_values = 1 / np.array(snr_values)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    ax1.plot(inv_snr_values, avg_estimates, color='r')
    ax1.fill_between(inv_snr_values,
                     np.array(avg_estimates) - np.sqrt(variances),
                     np.array(avg_estimates) + np.sqrt(variances),
                     color='red', alpha=0.2)
    ax1.set_xscale('log')
    ax1.set_xlim([inv_snr_values.min(), inv_snr_values.max()])
    ax1.set_ylim([0, fs_target / 2])
    ax1.set_title('Average frequency estimate for "Zauberflöte_vocal.wav"')
    ax1.set_ylabel('Frequency Estimate (Hz)')
    ax1.grid(True)

    ax2.plot(inv_snr_values, avg_errors, color='b')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlim([inv_snr_values.min(), inv_snr_values.max()])
    ax2.set_ylim([1e-1, 1e4])
    ax2.set_title('Average error for "Zauberflöte_vocal.wav"')
    ax2.set_xlabel('1/SNR')
    ax2.set_ylabel('Error (Hz)')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


test_note()