import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.fft import fft, ifft

def Boll_base(señal, fs, long=50, umbral=8):
    """
    

    Parametros
    ----------
    señal : Array
        Señal a aplicar el algoritmo.
    fs : Int
        Frecuncia de muestreo.
    long : Int, opcional
        Longitud de la ventana en ms. Predeterminado en 50 ms.
    umbral: Int, opcional
        Parámetro para establecer el valor del ruido a filtrar. Predeterminado en 10.
    Salida:
    Array procesado
    """
    
    #Conversión de ventana de tiempo a muestras
    cant_muestras_ventana = int(np.round(long*fs/1000,0))
    cant_ventanas = int(len(señal)/(cant_muestras_ventana/2)-1)
    
    #Se ventanea la señal y se agrega a una matriz
    
    array_ventaneo = np.zeros((cant_ventanas,cant_muestras_ventana))
    overlap = int(cant_muestras_ventana/2)
    
    array_hanning = np.zeros((cant_ventanas,cant_muestras_ventana))
    n = np.arange(0,cant_muestras_ventana)
    ventana_hanning = 0.5 - 0.5*np.cos((2*np.pi*n)/(cant_muestras_ventana-1))
    
    array_fft = np.zeros((cant_ventanas,cant_muestras_ventana), dtype=complex)
    
    array_hamming = np.zeros((cant_ventanas,cant_muestras_ventana))
    ventana_hamming = 0.54 - 0.46*np.cos((2*np.pi*n)/(cant_muestras_ventana-1))
    
    salida_ste = np.zeros(cant_ventanas)
    
    for i in range(0,cant_ventanas):        
        array_ventaneo[i] = señal[i*overlap:cant_muestras_ventana+i*overlap]
        array_hanning[i] = array_ventaneo[i]*ventana_hanning
        array_fft[i] = fft(array_hanning[i])
        array_hamming[i] = array_ventaneo[i]*ventana_hamming
        salida_ste[i] = np.sum(abs(array_hamming[i])**2)/cant_muestras_ventana
    
    
    
    # #Se aplica la ventana de Hanning a cada ventana
    # array_hanning = np.zeros((cant_ventanas,cant_muestras_ventana))
    # n = np.arange(0,cant_muestras_ventana)
    # ventana_hanning = 0.5 - 0.5*np.cos((2*np.pi*n)/(cant_muestras_ventana-1))
    # for i in range(0,cant_ventanas):
    #     array_hanning[i] = array_ventaneo[i]*ventana_hanning

    # #Se aplica FFT a cada una de las ventanas
    # array_fft = np.zeros((cant_ventanas,cant_muestras_ventana), dtype=complex)
    # for i in range(0,cant_ventanas):
    #     array_fft[i] = fft(array_hanning[i])
    
    
    
    # array_hamming = np.zeros((cant_ventanas,cant_muestras_ventana))
    # ventana_hamming = 0.54 - 0.46*np.cos((2*np.pi*n)/(cant_muestras_ventana-1))
    # for i in range(0,cant_ventanas):
    #     array_hamming[i] = array_ventaneo[i]*ventana_hamming
    
    # salida_ste = np.zeros(cant_ventanas)
    # for i in range(0,cant_ventanas):
    #     salida_ste[i] = np.sum(abs(array_hamming[i])**2)/cant_muestras_ventana
    
    valor_ref = umbral*np.min(salida_ste)
    
    # array_sin_habla = array_hanning
    # for i in range(len(salida_ste)):
    #     if salida_ste[i] > valor_ref:
    #         array_sin_habla[i] = np.zeros(cant_muestras_ventana)
    
    array_sin_habla = np.array([])
    for i in range(len(salida_ste)):
        if salida_ste[i] < valor_ref:          
            array_sin_habla = np.append(array_sin_habla, abs(array_fft[i])) #Se crea un array de ffts del ruido ventaneado
    
    mu = np.mean(array_sin_habla)
    
    spectral_estimator = (1-mu/(abs(array_fft)))*array_fft
    







import_s = sf.read('ClarinetHiNoise.wav')
signal = import_s[0]
fs = import_s[1]
signal_L = np.transpose(signal)[0]
Boll_base(signal_L, fs)