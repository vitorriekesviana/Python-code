import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.fft import fft
from scipy.ndimage import uniform_filter1d
import os

# Replace this with the full path to your .h5 file
file_name = r"file_path"

# Open HDF5 file
with h5py.File(file_name, 'r') as f:
    data_group = f['Recording_0/AnalogStream_0']
    matrix_full = np.array(data_group['ChannelData']).astype(np.float64)
    duration = f['Recording_0'].attrs['Duration'][0].astype(np.float64)
    time_s = duration / 1e6

# Extract relevant parts for saving file
parts = file_name.split(os.sep)
part1 = parts[5]
part2 = parts[7]
desired_path = f"{part1}/{part2}"
print("Desired path:", desired_path)

# Sampling frequency
num_samples_full = matrix_full.shape[1]
sampling_frequency = int(round(num_samples_full / time_s))
print("Sampling frequency:", sampling_frequency)

# Extract desired time segment
tempo_initial = 120
tempo_final = 130
x = int(round(tempo_initial * sampling_frequency))
y = int(round(tempo_final * sampling_frequency))
data_analysis = matrix_full[:, x:y]

# Signal processing parameters
num_channels_reduced, num_samples_reduced = data_analysis.shape
Fs = sampling_frequency
T = 1 / Fs
L = num_samples_reduced if num_samples_reduced % 2 == 0 else num_samples_reduced - 1
t = np.arange(0, L) * T

# Bandpass filter
f_low = 1
f_high = 4
order = 2
data_analysis1 = np.zeros_like(data_analysis)
Wn = [f_low / (Fs / 2), f_high / (Fs / 2)]
b, a = butter(order, Wn, btype='band')

for i in range(num_channels_reduced):
    data_analysis1[i, :] = filtfilt(b, a, data_analysis[i, :])

# Plot magnitude spectrum for all channels
plt.figure()
colors = plt.cm.hsv(np.linspace(0, 1, num_channels_reduced))
for i in range(num_channels_reduced):
    X = fft(data_analysis1[i, :L])
    P2 = np.abs(X / L)
    P1 = P2[:L // 2 + 1]
    P1[1:-1] *= 2
    f = Fs * np.arange(0, L // 2 + 1) / L
    plt.plot(f, P1, color=colors[i])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Magnitude Spectrum of All Channels')
plt.xlim([0, 10])
plt.grid(True)
plt.show()

# Average magnitude spectrum
magnitude_spectra = np.zeros((num_channels_reduced, L // 2 + 1))
for i in range(num_channels_reduced):
    X = fft(data_analysis1[i, :L])
    P2 = np.abs(X / L)
    P1 = P2[:L // 2 + 1]
    P1[1:-1] *= 2
    magnitude_spectra[i, :] = P1

avg_magnitude_spectrum = magnitude_spectra.mean(axis=0)
avg_magnitude_spectrum_smoothed = uniform_filter1d(avg_magnitude_spectrum, size=10)
avg_magnitude_spectrum_smoothed_dB = 20 * np.log10(avg_magnitude_spectrum_smoothed)

# Plot average magnitude spectrum
plt.figure()
plt.plot(f, avg_magnitude_spectrum_smoothed_dB, 'r')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.title('Average Magnitude Spectrum')
plt.xlim([0, 10])
plt.ylim([60, 165])
plt.grid(True)
plt.show()

# Save to file
output_file = f"{desired_path}.txt"
print(f"Saving to: {output_file}")
np.savetxt(output_file, avg_magnitude_spectrum_smoothed_dB)
