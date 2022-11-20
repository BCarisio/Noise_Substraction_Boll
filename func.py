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
    
    if cant_muestras_ventana%2==1:               # Se hace par la cantidad de muestras de ventana para un overlap simétrico
        cant_muestras_ventana += 1
    
    cant_ventanas = int(len(señal)/(cant_muestras_ventana/2)-1)
    
    #Se ventanea la señal y se agrega a una matriz
    
    array_ventaneo = np.zeros((cant_ventanas,cant_muestras_ventana))        # Matriz de ventanas de la señal (f: cant. ventanas, c: cant. muestras)
    
    overlap = int(cant_muestras_ventana/2)
    
    # Definición ventana de Hanning ()
    
    array_hanning = np.zeros((cant_ventanas,cant_muestras_ventana))
    
    n = np.arange(0,cant_muestras_ventana)
    
    ventana_hanning = 0.5 - 0.5*np.cos((2*np.pi*n)/(cant_muestras_ventana-1))
    
    
    array_fft = np.zeros((cant_ventanas,cant_muestras_ventana), dtype=complex)
    
    
    # Definición ventana de Hamming (para estimación del STE)
    
    array_hamming = np.zeros((cant_ventanas,cant_muestras_ventana))
    
    ventana_hamming = 0.54 - 0.46*np.cos((2*np.pi*n)/(cant_muestras_ventana-1))
    
    salida_ste = np.zeros(cant_ventanas) # Data del STE de cada ventana de la señal
    
    for i in range(0,cant_ventanas):        
        
        array_ventaneo[i] = señal[i*overlap:cant_muestras_ventana+i*overlap]        # Ventaneo de la señal con la superposición especificada
        array_hanning[i] = array_ventaneo[i]*ventana_hanning                        # Aplicación de ventana de Hanning
        array_fft[i] = fft(array_hanning[i])                                        # Array de las fft de cada ventana de la señal    
        
        # STE
        
        array_hamming[i] = array_ventaneo[i]*ventana_hamming                        # Por otro lado, se aplica la ventana de Hamming a las ventanas.
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
    
    valor_ref = umbral*np.min(salida_ste)             # Valor de referencia del STE a partir del cual se considera presencia de habla
    
    # array_sin_habla = array_hanning
    # for i in range(len(salida_ste)):
    #     if salida_ste[i] > valor_ref:
    #         array_sin_habla[i] = np.zeros(cant_muestras_ventana)
    
    array_sin_habla = np.array([])
    
    for i in range(len(salida_ste)):                   # Se crea un array de ffts del ruido ventaneado
        if salida_ste[i] < valor_ref:          
            array_sin_habla = np.append(array_sin_habla, abs(array_fft[i]))     # Array de fft del ruido
    
    
    mu = np.mean(array_sin_habla)                      # Esperanza de la magnitud espectral del ruido
    
    spectral_estimator = (1-mu/(abs(array_fft)))*array_fft         # Spectral Subtraction Estimator
    
    
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
    
    array_sin_habla_rnr = np.zeros(cant_muestras_ventana, dtype=complex)   # Fila de ceros para establecer dimensión
    
    for i in range(len(salida_ste)):            # Se crea un array sin habla de la señal procesada
        if salida_ste[i] < valor_ref:          
            array_sin_habla_rnr = np.vstack([array_sin_habla_rnr, spectral_estimator_hwr[i]])
    
    array_sin_habla_rnr = array_sin_habla_rnr[1:]           # Se elimina la primera fila de ceros
    
    phi = np.angle(array_sin_habla_rnr)      # Fase phi del ruido procesado
    
    Nr = array_sin_habla_rnr - mu*np.exp(1j*phi)                       # Noise Residual
    
    Nrmax = np.max(abs(Nr),axis=0)                    # Array de valores máximos del Nr por frecuencia
    
    spectral_estimator_rnr = np.zeros((cant_ventanas,cant_muestras_ventana),dtype=complex)   
    
    for i in range(cant_ventanas):              # Esquema de Residual Noise Reduction. Cuando corresponde, se asigna magnitud (nueva) y fase (del S post HWR)
        
        for w in range(cant_muestras_ventana):
            if i==0:                            # 1er ventana. Se compara entre la actual y la siguiente.
                
                if abs(spectral_estimator_hwr[i][w])>Nrmax[w]:
                    
                    spectral_estimator_rnr[i][w] = spectral_estimator_hwr[i][w]
                    
                else:
                    
                    spectral_estimator_rnr[i][w] = np.min([abs(spectral_estimator_hwr[i][w]),abs(spectral_estimator_hwr[i+1][w])])*np.exp(1j*np.angle(spectral_estimator_hwr[i][w]))
             
            elif i==cant_ventanas-1:              # Última ventana. Se compara entre la actual y la anterior.
                
                if abs(spectral_estimator_hwr[i][w])>Nrmax[w]:
                    
                    spectral_estimator_rnr[i][w] = spectral_estimator_hwr[i][w]
                    
                else:
                    
                    spectral_estimator_rnr[i][w] = np.min([abs(spectral_estimator_hwr[i-1][w]),abs(spectral_estimator_hwr[i][w])])*np.exp(1j*np.angle(spectral_estimator_hwr[i][w]))
                
            else:                               # Comparación entre valor de ventana actual, anterior y posterior.
                
                if abs(spectral_estimator_hwr[i][w])>Nrmax[w]:
                    
                    spectral_estimator_rnr[i][w] = spectral_estimator_hwr[i][w]
                    
                else:
                    
                    spectral_estimator_rnr[i][w] = np.min([abs(spectral_estimator_hwr[i-1][w]),abs(spectral_estimator_hwr[i][w]),abs(spectral_estimator_hwr[i+1][w])])*np.exp(1j*np.angle(spectral_estimator_hwr[i][w]))


    # Additional Signal Attenuation During Nonspeech Activity
    
    T = np.zeros(cant_ventanas,dtype=complex)
    
    for i in range(cant_ventanas):
         
        T[i] = 20*np.log10(np.mean(np.abs(array_fft[i]/mu)))         # Cálculo del parámetro de comparación T, por ventana, para detectar ausencia de habla
    
    
    spectral_estimator_att = np.zeros(np.shape(spectral_estimator_rnr),dtype=complex)

    for i in range(0,cant_ventanas):                    # Asignación de atenuación por ventanas según su valor de T
        
        if T[i] >= -12:
            spectral_estimator_att[i] = spectral_estimator_rnr[i]
        
        else:
            spectral_estimator_att[i] = (10**(-30/20))*spectral_estimator_rnr[i]

    
    # print(spectral_estimator_att,np.shape(spectral_estimator_att))
    
    return spectral_estimator_att
    
    

def synthesis(x,fs):
    
    # Synthesis
    
    array_ifft = np.zeros(np.shape(x),dtype=complex)
    
    long_ventana = np.shape(x)[1]
    
    cant_ventanas = np.shape(x)[0]
    
    for i in range(0,cant_ventanas):     # Antitransformada de las DFT's ingresadas de la señal particular
        
        array_ifft[i] = ifft(x[i])
        
    
    señal_filtrada = np.zeros(int(long_ventana*(cant_ventanas+1)/2),dtype=complex)
    
    for i in range(cant_ventanas):             # Compensación de superposición entre ventanas
        
        if i==cant_ventanas:
            
            señal_filtrada
    
        señal_filtrada[i*int(long_ventana/2):i*int(long_ventana/2)+long_ventana] += array_ifft[i]
    
    print(señal_filtrada,np.shape(señal_filtrada))
    
    return señal_filtrada
    

#%%
import_s = sf.read('ClarinetHiNoise.wav')
signal = import_s[0]
fs = import_s[1]
signal_L = np.transpose(signal)[0]

BB = Boll_base(signal_L, fs)

#%%
synthesis(BB,fs)

sf.write('Salida2.wav',np.real(synthesis(BB,fs))/np.max(np.real(synthesis(BB,fs))),fs)
