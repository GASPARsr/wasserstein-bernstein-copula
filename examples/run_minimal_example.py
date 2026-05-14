
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.stats import wasserstein_distance
from sklearn.metrics import mean_squared_error

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from bernstein_simulation import preparar_Bernstein_cond, simular_desde_uu
from objective_function import funcion_objetivo
import variogram as cve


# Fixed seed for reproducibility
np.random.seed(42)

# Load benchmark dataset
data_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "Datos_hess_400_datos.csv"
)

df = pd.read_csv(data_path)

# Expected columns from benchmark
x = df["x_utm"].values
y = df["y_utm"].values
h = df["h"].values
K = df["K"].values
cont = df["contadores"].values

# Known points
mask = cont == 1
xk, yk = x[mask], y[mask]
Kk = K[mask]
hk = h[mask]

coords = np.column_stack((x, y))
tree = KDTree(np.column_stack((xk, yk)))

# Reference experimental variogram from known hydraulic conductivity values
_, gamma_ref = cve.calcular_variograma_experimental(
    xk,
    yk,
    Kk,
    lag_size=236,
    N_lags=15
)

# Local Bernstein copula precomputation
precomp = preparar_Bernstein_cond(
    coords_Todos=coords,
    N_vecinos=6,
    h_conocidos=hk,
    K_conocidos=Kk,
    h=h,
    tree=tree,
    tolerancia=1e-4
)

# Generate one reproducible conditional realization
uu = np.random.uniform(0, 1, (1, coords.shape[0]))
sim = simular_desde_uu(precomp, uu)[0]

# Evaluate tri-objective function
score = funcion_objetivo(
    sim,
    xk,
    yk,
    Kk,
    x,
    y,
    None,
    236,
    gamma_ref,
    1 / 3,
    1 / 3,
    1 / 3
)

# Additional diagnostic metrics at known locations
sim_known = sim[:len(Kk)]
interp_error = np.sqrt(mean_squared_error(Kk, sim_known))
wass = wasserstein_distance(Kk, sim_known)

print("Simulation completed successfully")
print(f"Objective score: {score:.6f}")
print(f"Interpolation error: {interp_error:.6f}")
print(f"Wasserstein distance: {wass:.6f}")
