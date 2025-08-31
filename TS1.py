# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 18:36:22 2025
@author: Matias Carbajal
"""
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt 

# --------------------
# Parámetros
# --------------------
N = 250             # número de muestras
f = 2000            # frecuencia de la señal (Hz)
fs = 50000          # frecuencia de muestreo (Hz)
fsquare = 4000      # frecuencia señal cuadrada

# Vector de tiempo
t = np.arange(N) / fs

# --------------------
# Señales
# --------------------
s1 = np.sin(2*np.pi*f*t)                # senoide 2 kHz
s2 = 2 * np.sin(2*np.pi*f*t + np.pi/2)  # misma senoide, x2 y desfase π/2
s3 = np.sin(2*np.pi*(f/2)*t)            # senoide 1 kHz (moduladora)
s4 = s1 * s3                             # AM
sq = signal.square(2 * np.pi * fsquare * t)  # señal cuadrada 4 kHz
A = np.max(np.abs(s4)) * 0.75
s5 = np.clip(s4, a_min=-A, a_max=A)     # señal recortada 75%

# Pulso rectangular
Tp = 0.01                # duración del pulso 10 ms
N_pulso = int(Tp * fs)
N_total = 1000
t_pulso = np.arange(N_total)/fs
pulso = np.zeros(N_total)
pulso[:N_pulso] = 1
E_pulso = np.sum(pulso**2)
print(f"Pulso rectangular 10 ms: N = {N_total}, E = {E_pulso}")

# --------------------
# Gráficas de señales
# --------------------
def plot_signal(t, sig, title, ylabel="Amplitud [V]"):
    plt.figure()
    plt.plot(t, sig)
    plt.title(title)
    plt.xlabel("Tiempo [s]")
    plt.ylabel(ylabel)
    plt.grid(True)

plot_signal(t, s1, "Señal senoide 2 kHz")
plot_signal(t, s2, "Señal senoide 2 kHz amplificada y desfasada π/2")
plot_signal(t, s3, "Señal senoide 1 kHz (moduladora)")
plot_signal(t, s4, "Señal AM (2 kHz modulada por 1 kHz)")
plot_signal(t, s5, "Señal recortada (75% amplitud máxima)")
plot_signal(t, sq, "Señal cuadrada 4 kHz")
plot_signal(t_pulso, pulso, "Pulso rectangular de 10 ms")

# --------------------
# Autocorrelación de s1
# --------------------
R_s1 = np.correlate(s1, s1, mode='full') / len(s1)
lags = np.arange(-len(s1)+1, len(s1))

plt.figure()
plt.plot(lags, R_s1, label="Autocorrelación s1")
plt.title("Autocorrelación de s1 (senoide 2 kHz)")
plt.xlabel("Desplazamiento [muestras]")
plt.ylabel("R_s1[k]")
plt.grid(True)
plt.legend()

# --------------------
# Correlaciones cruzadas
# --------------------
signals = [s2, s3, s4, s5, sq]
names = ["s2", "s3", "s4", "s5", "sq"]

plt.figure()
for sig, name in zip(signals, names):
    R = np.correlate(s1, sig, mode="full") / len(s1)
    plt.plot(np.arange(-len(s1)+1, len(s1)), R, label=f"s1 vs {name}")

plt.title("Correlaciones cruzadas de s1 con otras señales")
plt.xlabel("Desplazamiento [muestras]")
plt.ylabel("R_s1,sig[k]")
plt.grid(True)
plt.legend()

# --------------------
# Verificación identidad trigonométrica
# --------------------
alpha = 2*np.pi*f*t
beta = 2*np.pi*(f/2)*t
lhs = 2 * np.sin(alpha) * np.sin(beta)
rhs = np.cos(alpha - beta) - np.cos(alpha + beta)

plt.figure()
plt.plot(t, lhs, label="2 sin(α) sin(β)")
plt.plot(t, rhs, '--', label="cos(α-β) - cos(α+β)")
plt.title("Verificación de la identidad trigonométrica")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.grid(True)
plt.legend()

plt.show()
