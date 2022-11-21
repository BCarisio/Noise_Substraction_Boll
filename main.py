from func import Boll_base, Boll_alt, synthesis, STE, ZCR, energia, estimacion_max_ruido
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

#%%
# Se importa la señal de referencia

# import_s = sf.read('ClarinetHiNoise.wav')
# import_s = sf.read('VegaHiNoise.wav')
import_s = sf.read('GlockHiNoise.wav')

señal = import_s[0]
fs = import_s[1]
señal_L = np.transpose(señal)[0]        # Se toma un solo canal de la señal estéreo

# Aplicación de ambos métodos y sus síntesis.

BB = Boll_base(señal_L, fs)

señal_BB = synthesis(BB)

BA = Boll_alt(señal_L, fs)

señal_BA = synthesis(BA)
#%%
# Creación de archivos .wav con los resultados/señales filtradas mediante cada método.

sf.write('Salida_BB_Glock.wav',np.real(señal_BB)/np.max(np.real(señal_BB)),fs)

sf.write('Salida_BA_Glock.wav',np.real(señal_BA)/np.max(np.real(señal_BA)),fs)

#%%
# Análisis y comparación de los STE y ZCR de ambos métodos

STE_señal = STE(señal_L)
ZCR_señal = ZCR(señal_L)

STE_BB = STE(señal_BB)
ZCR_BB = ZCR(señal_BB)

STE_BA = STE(señal_BA)
ZCR_BA = ZCR(señal_BA)

plt.figure(figsize=(15,6))
plt.suptitle("Filtrado con método de Boll Básico")

plt.subplot(1,2,1)
plt.plot(STE_BB/np.max(STE_BB))
plt.plot(STE_señal/np.max(STE_señal))
plt.ylabel("Magnitud")
plt.xlabel("Ventana")
plt.title("Short Time Energy")

plt.subplot(1,2,2)
plt.plot(ZCR_BB/np.max(ZCR_BB))
plt.plot(ZCR_señal/np.max(ZCR_señal))
plt.ylabel("Magnitud")
plt.xlabel("Ventana")
plt.title("Zero-Crossing Rate")

plt.tight_layout()
plt.show()

plt.figure(figsize=(15,6))
plt.suptitle("Filtrado con método de Boll Alternativo (Sobresubstracción)")

plt.subplot(1,2,1)
plt.plot(STE_BA/np.max(STE_BA))
plt.plot(STE_señal/np.max(STE_señal))
plt.ylabel("Magnitud")
plt.xlabel("Ventana")
plt.title("Short Time Energy")

plt.subplot(1,2,2)
plt.plot(ZCR_BA/np.max(ZCR_BA))
plt.plot(ZCR_señal/np.max(ZCR_señal))
plt.ylabel("Magnitud")
plt.xlabel("Ventana")
plt.title("Zero-Crossing Rate")

plt.tight_layout()
plt.show()

#%%
# Análisis de "Relación de Energía" para la señal original, la señal_BB y la señal_BA

# import_s_limpia = sf.read('ClarinetREF.wav')
# import_s_limpia = sf.read('VegaREF.wav')
# import_s_limpia = sf.read('GlockREF.wav')

señal_limpia = import_s_limpia[0]
señal_L_limpia = np.transpose(señal_limpia)[0]

# El array de "resta" de cada caso corresponde al ruido resultante de la resta entre señal total y la señal limpia de cada caso.

resta_original = señal_L - señal_L_limpia

resta_BB = señal_BB - señal_L_limpia

resta_BA = señal_BA - señal_L_limpia

plt.figure(figsize=(15,6))

# plt.plot(resta_original/np.max(resta_original),label="Original")
# plt.plot(resta_BB/np.max(resta_BB),label="Boll Básico")
# plt.plot(resta_BA/np.max(resta_BA),label="Boll Alternativo")
plt.plot(resta_original,label="Original")
plt.plot(resta_BB,label="Boll Básico")
plt.plot(resta_BA,label="Boll Alternativo")
plt.ylabel("Magnitud")
plt.xlabel("Muestras")
plt.title("Ruido Resultante")

plt.legend()

plt.tight_layout()
plt.show()

# Energía (se calculan y comparan los porcentajes de la energía de la señal total que representa cada ruido resultante)

per_e_original = 100*energia(resta_original) / energia(señal_L)
per_e_BB = 100*energia(resta_BB) / energia(señal_BB)
per_e_BA = 100*energia(resta_BA) / energia(señal_BA)

print("Porcentaje de energía del ruido resultante de la señal original:", per_e_original)
print("Porcentaje de energía del ruido resultante de la señal original:", per_e_BB)
print("Porcentaje de energía del ruido resultante de la señal original:", per_e_BA)

#%%

# SNR

SNR_original = np.max(señal_L)/estimacion_max_ruido(señal_L)
SNR_BB = np.max(señal_BB)/estimacion_max_ruido(señal_BB)
SNR_BA = np.max(señal_BA)/estimacion_max_ruido(señal_BA)

print("SNR para señal original:",np.round(SNR_original,2))
print("SNR para señal Boll Básico:",np.round(np.real(SNR_BB),2))
print("SNR para señal Boll Alternativo:",np.round(np.real(SNR_BA),2))