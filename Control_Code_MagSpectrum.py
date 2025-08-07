import numpy as np
import matplotlib.pyplot as plt

# ----- PARAMETERS -----
num_channels = 60               # Total number of sine waves
num_samples = 60                # Samples per wave
Fs = 200                        # Sampling frequency (Hz)
T = 1 / Fs                      # Sampling period
L = num_samples                 # Signal length
t = np.arange(L) * T            # Time vector
frequency_range = np.linspace(1, 60, num_channels)  # 1–60 Hz range

# ----- USER INPUT -----
visualize = int(input("How many frequency signals do you want to visualize? (1–60): "))
if visualize < 1 or visualize > 60:
    raise ValueError("Please enter a value between 1 and 60.")

# ----- SINE WAVE GENERATION -----
matrix_control = np.zeros((num_channels, num_samples))
for i in range(num_channels):
    f = frequency_range[i]
    matrix_control[i, :] = np.sin(2 * np.pi * f * t)

# ----- PLOT: MAGNITUDE SPECTRA -----
plt.figure()
colors = plt.cm.hsv(np.linspace(0, 1, visualize))
for i in range(visualize):
    X = np.fft.fft(matrix_control[i, :])
    P2 = np.abs(X / L)
    P1 = P2[:L//2 + 1]
    P1[1:-1] *= 2
    f = Fs * np.arange(0, L//2 + 1) / L
    plt.plot(f, P1, color=colors[i])

plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title(f"Magnitude Spectrum of First {visualize} Channels")
plt.xlim([0, 80])
plt.grid(True)
plt.tight_layout()

# ----- PLOT: AVERAGE MAGNITUDE SPECTRUM (in dB) -----
magnitude_spectra = np.zeros((visualize, L//2 + 1))
for i in range(visualize):
    X = np.fft.fft(matrix_control[i, :])
    P2 = np.abs(X / L)
    P1 = P2[:L//2 + 1]
    P1[1:-1] *= 2
    magnitude_spectra[i, :] = P1

avg_magnitude_spectrum = np.mean(magnitude_spectra, axis=0)
avg_magnitude_spectrum_dB = 20 * np.log10(avg_magnitude_spectrum + 1e-12)

f = Fs * np.arange(0, L//2 + 1) / L
plt.figure()
plt.plot(f, avg_magnitude_spectrum_dB, 'r', linewidth=2)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title(f"Average Magnitude Spectrum of First {visualize} Channels")
plt.xlim([0, 80])
plt.grid(True)
plt.tight_layout()

# ----- PLOT: TIME-DOMAIN SIGNALS -----
plt.figure()
for i in range(visualize):
    plt.plot(t, matrix_control[i, :], color=colors[i])

plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title(f"Time-Domain Sine Waves of First {visualize} Channels")
plt.grid(True)
plt.tight_layout()

plt.show()
