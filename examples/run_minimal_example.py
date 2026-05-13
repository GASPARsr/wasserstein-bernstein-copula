
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from bernstein_simulation import preparar_Bernstein_cond, simular_desde_uu
from objective_function import funcion_objetivo
import variogram as cve

df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'Datos_hess_400_datos.csv'))
# Expected columns from benchmark
x = df['x_utm'].values
y = df['y_utm'].values
h = df['h'].values
K = df['K'].values
cont = df['contadores'].values

mask = cont == 1
xk, yk = x[mask], y[mask]
Kk = K[mask]
hk = h[mask]

coords = np.column_stack((x, y))
tree = KDTree(np.column_stack((xk, yk)))

_, gamma_ref = cve.calcular_variograma_experimental(xk, yk, Kk, lag_size=236, N_lags=15)

precomp = preparar_Bernstein_cond(coords, 6, hk, Kk, h, tree, 1e-4)
uu = np.random.uniform(0,1,(1, coords.shape[0]))
sim = simular_desde_uu(precomp, uu)[0]

score = funcion_objetivo(sim, xk, yk, Kk, x, y, None, 236, gamma_ref, 1/3, 1/3, 1/3)
print("Objective score:", score)
