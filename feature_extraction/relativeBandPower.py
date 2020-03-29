# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 18:15:33 2020

@author: Camila
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
sns.set(font_scale=1.2)
from scipy.integrate import simps

y1 =T4
# Passing from ndarray to array 
y1_s = y1.ravel()

# Define sampling frequency and time vector
time = np.arange(y1_s.size) / sf

# Plot the signal
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
plt.plot(time, y1, lw=1.5, color='k')
plt.xlabel('Tiempo (segundos)')
plt.ylabel('Voltaje (uV)')
plt.xlim([time.min(), time.max()])
plt.title('EEG Paciente sano. Canal' + 'F7')
sns.despine()


# Define window length (4 seconds)
win = 4 * sf
freqs, psd = signal.welch(y1_s, sf, nperseg=win)

# Plot the power spectrum
sns.set(font_scale=1.2, style='white')
plt.figure(figsize=(8, 4))
plt.plot(freqs, psd, color='k', lw=2)
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Densidad espectral de potencia (V^2 / Hz)')
plt.ylim([0, psd.max() * 1.1])
plt.title("Periodograma de Welch")
plt.xlim([0, freqs.max()])
sns.despine()


# Define lower and upper limits
low, high = 0.5, 4

idx_band = np.logical_and(freqs >= low, freqs <= high)

# Frequency resolution
freq_res = freqs[1] - freqs[0]  # = 1 / 4 = 0.25

## Compute the absolute power by approximating the area under the curve
band_power = simps(psd[idx_band], dx=freq_res)
print('Absolute delta power: %.3f uV^2' % band_power)

## Relative delta power (expressed as a percentage of total power)
total_power = simps(psd, dx=freq_res)
band_rel_power = band_power / total_power
print('Relative delta power: %.3f' % band_rel_power)

## Plot the power spectral density and fill the delta area
plt.figure(figsize=(7, 4))
plt.plot(freqs, psd, lw=2, color='k')
plt.fill_between(freqs, psd, where=idx_band, color='skyblue')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Densidad espectral de potencia (V^2 / Hz)')
plt.xlim([0, 10])
plt.ylim([0, psd.max() * 1.1])
plt.title("Periodograma de Welch")
sns.despine()

# This is getting really interesting: Here we calculate all the relative power
# bands in the channel

bands = {
        'alpha' : {
                'low': 8 ,
                'high': 13
                },
        'betha' : {
                'low': 13,
                'high': 30
                },
        'delta' : {
                'low': 0,
                'high': 4
                },
        'gamma' : {
                'low': 30,
                'high': 100
                },
        'theta' : {
                'low': 4,
                'high': 8
                },
        }

total = 0;

for i in bands:
    idx_band = np.logical_and(freqs >= bands[i]['low'], freqs <= bands[i]['high'])
    freq_res = freqs[1] - freqs[0]
    band_power = simps(psd[idx_band], dx=freq_res)
    total_power = simps(psd, dx=freq_res)
    band_rel_power = band_power / total_power
    print(band_rel_power)
    total += band_rel_power

print(total)
    

