def funcion_objetivo(simulacion, x_conocidos, y_conocidos, K_conocidos, x, y, K_desconocidos_real, lag_value, gamma_reales, w1, w2, w3):
    
    import numpy as np
    import variogram as cve
    from sklearn.metrics import mean_squared_error
    from scipy.stats import wasserstein_distance

    # Estandarización de la función objetivo 
    des_vario=np.std(gamma_reales, ddof=1)
    des_int=np.std(K_conocidos, ddof=1)

    # 1. Error del variograma
    _, gamma_sim = cve.calcular_variograma_experimental(
        x, y, simulacion[:], lag_value)
    mse_variograma = mean_squared_error(gamma_reales, gamma_sim)
    
    # 2. Error entre valores simulados y reales
    valores_simulados = simulacion[0:len(K_conocidos)]
    mse_valores = mean_squared_error(valores_simulados, K_conocidos)
    #print(mse_valores, w1/rango_vario * mse_variograma)
    #print(mse_variograma, w2/rango_datos_int * mse_valores)

    # 3. Mejor valor de wessernstein
    was_valores = 1-np.exp(-wasserstein_distance(valores_simulados, K_conocidos))
    
    # Distancia de Wasserstein entre las dos muestras
    
    # Función objetivo (ponderada)
    objetivo_total = (w1/des_vario) * np.sqrt(mse_variograma) + (w2/des_int) * np.sqrt(mse_valores) + w3 * was_valores
    return objetivo_total

def suma(a,b):
    c=a+b
    return c