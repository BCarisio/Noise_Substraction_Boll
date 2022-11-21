import numpy as np
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
    
    # Detección de momentos sin habla
    
    valor_ref = umbral*np.min(salida_ste)             # Valor de referencia del STE a partir del cual se considera presencia de habla
    
    array_sin_habla = np.array([])
    
    for i in range(len(salida_ste)):                   # Se crea un array de ffts del ruido ventaneado
        if salida_ste[i] < valor_ref:          
            array_sin_habla = np.append(array_sin_habla, abs(array_fft[i]))     # Array de fft del ruido
    
    
    mu = np.mean(array_sin_habla)                      # Esperanza de la magnitud espectral del ruido
    
    # spectral_estimator = (1-mu/(abs(array_fft)))*array_fft         # Spectral Subtraction Estimator
    
    
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
    
    return spectral_estimator_att
    
    
def synthesis(x):
    
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
    
    return señal_filtrada
    

def Boll_alt(señal, fs, long=50, umbral=8, alfa=5, beta=0.02):
    """
    Se aplica el método de Boll para substracción de ruido,
    utilizando la Substracción Espectral No Lineal.

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
    alfa: float, opcional
        Factor de sobre estimación (valores mayores o iguales a 1). Predeterminado en 5.
    beta: float, opcional
        Parámetro de piso espectral (valores entre 0 y 1). Predeterminado en 0.02.
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
    
    # Definición ventana de Hanning
    
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
    
    
    valor_ref = umbral*np.min(salida_ste)             # Valor de referencia del STE a partir del cual se considera presencia de habla
    
    
    array_sin_habla = np.array([])
    
    for i in range(len(salida_ste)):                   # Se crea un array de ffts del ruido ventaneado
        if salida_ste[i] < valor_ref:          
            array_sin_habla = np.append(array_sin_habla, abs(array_fft[i]))     # Array de fft del ruido
    
    
    mu = np.mean(array_sin_habla)                      # Esperanza de la magnitud espectral del ruido
    
    # spectral_estimator = (1-mu/(abs(array_fft)))*array_fft         # Spectral Subtraction Estimator
    
    
    # Spectral Substraction with Over Subtraction
    
    array_ss = np.zeros(np.shape(array_fft), dtype=complex) # Array de salida. Ventanas a las que se les aplica la sobre substracción.
    
    # Se define el promedio de estimación de ruido espectral
    
    D = np.zeros(cant_muestras_ventana, dtype=complex)   # Fila de ceros para establecer dimensión
    
    for i in range(len(salida_ste)):            # Se crea un array sin habla de la señal procesada
        if salida_ste[i] < valor_ref:          
            D = np.vstack([D, array_fft[i]])
            
    D = D[1:]           # Se elimina la primera fila de ceros
    
    # theta = np.angle(array_fft)      # Fase theta para aplicar al ruido 
    
    D = np.mean(D, axis=0)           # Promedios de ruido por frecuencia
    
    D_matriz = np.zeros(np.shape(array_fft), dtype=complex)
    
    for i in range(0,cant_ventanas):                # Matriz de promedios de magnitud de ruido de misma dimension que array_fft
        
        D_matriz[i] = D
    
    for  i in range(0,cant_ventanas):
        
        for w in range(0,cant_muestras_ventana):
            
            if abs(array_fft[i][w])**2 > (alfa+beta)*abs(D_matriz[i][w])**2:
                
                array_ss[i][w] = np.sqrt(abs(array_fft[i][w])**2 - abs(D_matriz[i][w])**2)
            
            else:
                
                array_ss[i][w] = np.sqrt(beta*abs(D_matriz[i][w])**2)
    
    # Half-Wave Rectification
    
    H = np.sqrt(1-(D_matriz/array_fft)**2) # Defino una función transferencia a partir de la salida y entrada de la sobre substracción
    
    Hr = (H + abs(H))/2                     # Defino el Hr con la mejora del 
    
    spectral_estimator_hwr = Hr*array_ss                       # S.E. post MA y HWR

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
    
    return spectral_estimator_att

# Framing Step

def framing (señal, duracion=2205, fs=44100, overlap=0):
    
    """
    In the short-term analysis or frame-based processing, 
    the audio signal is sliced into possibly overlapping N frames.
    
    señal: Array de señal de audio. (array)
    
    duracion: Duración del intervalo/frame en muestras.(int)
    (entre 10 y 50 ms) (2205 muestras son 50 ms para fs=44100)
    
    fs: Frecuencia de muestreo. (int)
    
    """
    
    x = señal                           # Señal de audio a ser dividida en N cuadros (frames)
    d = duracion                        # Longitud de cada uno de los N intervalos
    
    # Framing
    
    x_f = []                            # Array para los frames
    
    # x_f[0] = x[0:d]                     # Determinación del primer frame
    
    for i in range(0,int((len(x)-d+overlap)/d)):       # Asignación de los frames de la señal
        
        x_f.append(x[i*d-overlap:i*d-overlap+d])
        
        
    # Windowing (Hamming Window)
    
    n = np.arange(0,d)                          # Vector n muestras de longitud del frame
    
    w = 0.54 - 0.46 * np.cos(2*np.pi*n/d)       # Ventana de Hamming
    
    for i in range(0,len(x_f)):                 # Aplicación de ventaneo a cada frame
        
        x_f[i] = w*x_f[i]
        
    return x_f


# Short Time Energy

def STE(señal, d=2205):
    
    x = framing(señal,duracion=d)      # Aplico el framing a la señal x,  se obtiene la señal x separada en frames y el ancho d de frames
        
    STE = np.zeros(len(x))         
    
    for i in range(0,len(STE)):     # Cálculo del STE para cada frame i de la señal
        
        STE[i] = np.sum(np.abs((x[i]))**2)/d
    
    return STE


# Zero Crossing Rate ("Short Time Average Zero Crossing Rate")

def ZCR(señal, d=2205):
    
    """
    The ZCR show the level of noisiness of a given signal.
    In fact, noisy recorded audio signals correspond to
    high values of ZCR.
    """
    
    x = framing(señal,duracion=d)
      
    
    sgn = np.zeros((len(x),d))          # Array de ceros para definir la función "sign"
    
    for i in range(0,len(sgn)):      # Recorre cada frame de la señal dividida
    
        for n in range(0,d):         # Iteración en cada muestra de cada frame
        
            if x[i][n]>=0:           # Asignación de valores al array de sgn
                sgn[i][n] = 1
            
            else:
                sgn[i][n] = -1            
        
    ZCR = np.zeros(len(x))            
    
    for i in range(0,len(sgn)):
        num = []
        for n in range(0,d):
            num.append(np.abs(sgn[i][n]-sgn[i][n-1]))  # Armado del vector de restas entre un elemento i de sgn y su anterior
        
        ZCR[i] = np.sum(num)/(2*d)                     # Sumatoria de los resultados de todas las restas, dividido el doble del largo de la ventana     
        
    return ZCR
    

def energia(x):
    
    e = np.sum(np.abs(x)**2)
        
    return e
        

def estimacion_max_ruido(x):
    """
    Cálculo del máximo de las primeras 1000 muestras de una señal.
    La idea es tomar el valor máximo del ruido, para luego calcular el SNR de la señal.
    También se asume que la señal comienza solo con ruido.
    """
    return np.max(abs(x[0:1000]))