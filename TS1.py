# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 18:36:22 2025

@author: Augusto
"""
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt 

# Parámetros
N = 250              # número de muestras
f = 2000             # frecuencia de la señal (Hz)
fs = 50000          # frecuencia de muestreo (Hz)
fsquare=4000
# Vector de tiempo
t = np.arange(N) / fs

# Señales
s1 = np.sin(2*np.pi*f*t)                # senoide de 2 kHz
s2 = 2 * np.sin(2*np.pi*f*t + np.pi/2)  # misma senoide, amplificada x2 y desfase π/2
s3 = np.sin(2*np.pi*(f/2)*t)  
s4 = s3 * s1
sq = signal.square(2 * np.pi * fsquare * t)   #señal cuadrada
A = np.max(np.abs(s4)) * 0.75   # 75% de la amplitud máxima
s5 = np.clip(s4, a_min=-A, a_max=A)


#señal rectangular

Tp = 0.01                 # duración del pulso
N_pulso = int(Tp * fs)    # muestras del pulso
N_total = 1000            # total de muestras para ver pulso + ceros
t_pulso = np.arange(N_total)/fs

pulso = np.zeros(N_total)
pulso[:N_pulso] = 1       # pulso de 10 ms
E_pulso = np.sum(pulso**2)   # energía del pulso
print(f"Pulso rectangular 10 ms: N = {N_total}, E = {E_pulso}")
# Gráfico

plt.figure()
plt.plot(t, s1)
plt.title("Señal senoide 2 kHz")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid(True)

plt.figure()
plt.plot(t, s2)
plt.title("Señal senoide 2 kHz amplificada y desfasada π/2")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid(True)

plt.figure()
plt.plot(t, s3)
plt.title("Señal senoide 1 kHz (moduladora)")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid(True)

plt.figure()
plt.plot(t, s4)
plt.title("Señal AM (2 kHz modulada por 1 kHz)")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid(True)

plt.figure()
plt.plot(t, s5)
plt.title("Señal recortada")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid(True)

plt.figure()
plt.plot(t, sq)
plt.title("Señal cuadrada")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid(True)

plt.figure()
plt.plot(t_pulso, pulso)
plt.title("Pulso rectangular de 10 ms")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")

#EJERCICIO 2

signals = [s2, s3, s4, sq]
names = ["s2", "s3", "s4", "sq"]

print("\nVerificación de ortogonalidad con s1:")
umbral = 1e-10
for sig, name in zip(signals, names):
    inner = np.sum(s1 * sig)
    if abs(inner) < umbral:
        print(f"s1 y {name} → ORTOGONAL (producto interno ≈ {inner:.2e})")
    else:
        print(f"s1 y {name} → NO ORTOGONAL (producto interno = {inner:.4f})")
# EJERCICIO 3: Autocorrelación y correlación cruzada

# --- Autocorrelación de s1 ---
R_s1 = np.correlate(s1, s1, mode="full") / len(s1)   # normalizada
lags = np.arange(-len(s1)+1, len(s1))      # eje en muestras

plt.figure()
plt.plot(lags, R_s1, label="Autocorrelación s1")
plt.title("Autocorrelación de s1 (senoide 2 kHz)")
plt.xlabel("Desplazamiento [muestras]")
plt.ylabel("R_s1[k]")
plt.grid(True)
plt.legend()
plt.show()

# --- Correlaciones cruzadas de s1 con otras señales ---
signals = [s2, s3, s4, sq]   # las otras señales
names   = ["s2", "s3", "s4", "sq"]

plt.figure()
for sig, name in zip(signals, names):
    R = np.correlate(s1, sig, mode="full") / len(s1)   # normalizada
    plt.plot(lags, R, label=f"s1 vs {name}")

plt.title("Correlaciones cruzadas de s1 con otras señales")
plt.xlabel("Desplazamiento [muestras]")
plt.ylabel("R_s1,sig[k]")
plt.grid(True)
plt.legend()
plt.show()
    
#IDENTIDAD     
alpha = 2*np.pi*1000*t   # ejemplo: 1 kHz
beta  = 2*alpha          # el doble

lhs = 2*np.sin(alpha)*np.sin(beta)
rhs = np.cos(alpha-beta) - np.cos(alpha+beta)

plt.figure()
plt.plot(t, lhs, label="2sin(α)sin(β)")
plt.plot(t, rhs, '--', label="cos(α-β)-cos(α+β)")
plt.legend()
plt.title("Verificación de la identidad trigonométrica")
plt.grid(True)
plt.show()
    