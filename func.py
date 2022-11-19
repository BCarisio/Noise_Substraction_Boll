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
        Parámetro para establecer el valor del ruido a filtrar. Predeterminado en 8.
    Salida:
    Array procesado
    """
    
    #Conversión de ventana de tiempo a muestras
    cant_muestras_ventana = int(np.round(long*fs/1000,0))
    
    if cant_muestras_ventana%2==1:                          # Se hace par la cantidad de muestras de ventana para un overlap simétrico
        cant_muestras_ventana += 1
    
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
        salida_ste[i] = np.sum(abs(array_hamming[i])**2)/cant_muestras_ventana      # STE de la señal
    
    
    
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
    
    # Spectral Subtraction Estimator
    
    valor_ref = umbral*np.min(salida_ste)                   # Valor de referencia a partir del cual se considera presencia de habla
    
    # array_sin_habla = array_hanning
    # for i in range(len(salida_ste)):
    #     if salida_ste[i] > valor_ref:
    #         array_sin_habla[i] = np.zeros(cant_muestras_ventana)
    
    array_sin_habla = np.array([])
    
    for i in range(len(salida_ste)):                     # Se crea un array de ffts del ruido ventaneado
        if salida_ste[i] < valor_ref:          
            array_sin_habla = np.append(array_sin_habla, abs(array_fft[i])) 
    
    
    mu = np.mean(array_sin_habla)
    
    spectral_estimator = (1-mu/(abs(array_fft)))*array_fft
    
    
    # Magnitude Averaging
    
    promedio_magnitud = np.mean(array_fft,axis=0)       # Promedio de la magnitud de la fft por frecuencia de todas las ventanas.
    
    promedio_magnitud_full = np.zeros(np.shape(array_fft), dtype=complex)
    
    
    for i in range(0,cant_ventanas):                # Matriz de promedios de magnitud de misma dimension que array_fft
        
        promedio_magnitud_full[i] = promedio_magnitud

    # spectral_estimator_ma = (1-mu/promedio_magnitud_full)*array_fft      # Spectral Estimator hasta el proceso de Magnitud Averaging
    
    # Half-Wave Rectification
    
    Hr = ((1-mu/promedio_magnitud_full) + np.abs((1-mu/promedio_magnitud_full)))/2 # Defino el Hr con la mejora del MA
    
    spectral_estimator_hwr = Hr*array_fft                       # S.E. post MA y HWR

    # Residual Noise Reduction

    # for i in range(len(salida_ste)):            # Se crea un array sin habla de la señal procesada
    #     if salida_ste[i] < valor_ref:          
    #         array_sin_habla_rnr = np.append(array_sin_habla_rnr, abs(spectral_estimator_hwr[i])) 

    spectral_estimator_rnr = 0    

    # Additional Signal Attenuation During Nonspeech Activity
    
    T = 20*np.log10(np.mean(np.abs(spectral_estimator_rnr/mu)))         # Cálculo del parámetro de comparación T para detectar ausencia de habla

    spectral_estimator_att = np.zeros(np.shape(spectral_estimator_rnr))

    for i in range(0,cant_ventanas):                    # Asignación de atenuación por ventanas según el valor de T
        
        if T[i]>=-12:
            spectral_estimator_att[i] = spectral_estimator_rnr[i]
        
        else:
            spectral_estimator_att[i] = (10**(-30/20))*spectral_estimator_rnr[i]




import_s = sf.read('ClarinetHiNoise.wav')
signal = import_s[0]
fs = import_s[1]
signal_L = np.transpose(signal)[0]
Boll_base(signal_L, fs)