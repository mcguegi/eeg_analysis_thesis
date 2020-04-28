# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 20:37:34 2020

@author: Camila
"""
import scipy 
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------------------------------------------------------------------

# Plots of the five extracted 'special' channels
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5)
fig.suptitle('Canales seleccionados')
ax1.plot(F3, c='#E25FF0')
ax1.set_title('F3')
ax2.plot(C3, c='#003333')
ax2.set_title('C3')
ax3.plot(P3, c='#5FF9DA')
ax3.set_title('P3')
ax4.plot(T4, c='#CE591F')
ax4.set_title('T4')
ax5.plot(F7, c='#A6FF4B')
ax5.set_title('F7')
plt.tight_layout()


# Lectura de la señal y freq de muestreo
y1 = P3

# Figura en el dominio del tiempo
plt.plot(y1, c='red')
plt.grid()
plt.title ('Señal en el dominio de tiempo')
plt.xlabel ('Tiempo')
plt.ylabel ('Amplitud')


# Análisis de fourier
Y1 = fft(y1,axis=0)

# Escala de frecuencia
N = len(y1)
K = np.arange(0, N, 1)
f=sf*K/N

Y1abs = [abs(x) for x in Y1] 

# Figura en el dominio de la frecuencia
plt.plot(f,Y1abs, c='green')
plt.grid()
plt.title ('Señal en el dominio de frecuencia')
plt.xlabel ('Frecuencia (Hz)')
plt.ylabel ('Amplitud')

# Figura del espectro en frecuencia Wn
Wn = 2*K/N
plt.plot(Wn,Y1abs,c='blue')
plt.grid()
plt.title ('Señal en el dominio de ..')
plt.xlabel ('Frecuencia (Hz)')
plt.ylabel ('Amplitud')



sns.set(palette="deep")
f, axes = plt.subplots(5, 1, figsize=(7, 7), sharex=True)
sns.despine(left=True)

sns.lineplot(data=F3, ax=axes[0],legend=False).set_title('F3')
sns.lineplot(data=C3, ax=axes[1],legend=False).set_title('C3')
sns.lineplot(data=P3, ax=axes[2],legend=False).set_title('P3')
sns.lineplot(data=T4, ax=axes[3],legend=False).set_title('T4')
sns.lineplot(data=F7, ax=axes[4],legend=False).set_title('F7')

plt.setp(axes, yticks=[])
plt.tight_layout()