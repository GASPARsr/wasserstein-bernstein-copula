def calcular_variograma_experimental(X, Y, valores, lag_size=236, N_lags=15):
    import numpy as np
    n = len(X)
    matriz_distancias = np.zeros((n, n))
    a=np.c_[X[:],Y[:]]
    b=np.zeros((2,n))
    b[0,:],b[1,:] = X[:], Y[:]
    for i in range(n):
        matriz_distancias[i,:] = np.sqrt(np.sum((a[i,:,np.newaxis]-b)**2,axis=0))
    
    # Determinar la distancia máxima
    max_distancia = matriz_distancias.max()
    # Inicializar variables para el cálculo del variograma
    lag = 0
    siguiente_lag = lag_size
    hs = [0]
    gammas = []

    while (lag <= max_distancia) &  (len(hs)<N_lags+1):
    # while (lag <= max_distancia):
        pares2= np.where((matriz_distancias > lag) & (matriz_distancias <= siguiente_lag))
        if pares2[0].size>0:
            suma = np.sum((valores[pares2[0]]-valores[pares2[1]])**2)
            semivarianza = (1 / (2 * len(pares2[0]))) * suma
            gammas.append(semivarianza)
            hs.append(siguiente_lag)
        lag = siguiente_lag
        siguiente_lag += lag_size
   
    hs=hs[:N_lags]
    return np.array(hs), np.array(gammas)

def suma(a,b):
    c=a+b
    return (c)