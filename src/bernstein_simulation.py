# Simula_Bernstein.py

def preparar_Bernstein_cond(coords_Todos, N_vecinos, h_conocidos, K_conocidos, h, tree, tolerancia):
    import numpy as np
    from scipy.stats import binom

    n_puntos = coords_Todos.shape[0]

    # ------------------------------------------------------------
    # Vecinos y muestra local
    # ------------------------------------------------------------
    _, idx = tree.query(coords_Todos, k=N_vecinos)

    muestra = np.zeros((n_puntos, N_vecinos, 2))
    muestra[:, :, 0] = h_conocidos[idx]
    muestra[:, :, 1] = K_conocidos[idx]

    # ------------------------------------------------------------
    # Copula empírica local
    # ------------------------------------------------------------
    mat_copem = np.zeros((n_puntos, N_vecinos + 1, N_vecinos + 1))
    arange3 = np.arange(N_vecinos + 1) / N_vecinos
    mat_copem[:, N_vecinos, :] = arange3

    Mat_xyord = np.zeros_like(muestra)
    Y_ord = np.zeros((n_puntos, N_vecinos))

    for i in range(n_puntos):
        mat_xy = muestra[i, :, :]
        mat_ord = mat_xy[mat_xy[:, 0].argsort()]
        Mat_xyord[i, :, :] = mat_ord
        Y_ord[i, :] = np.sort(mat_ord[:, 1])

        for j in range(N_vecinos):
            columna = np.zeros(N_vecinos + 1)
            columna[1:] = (mat_ord[:, 1][j] <= Y_ord[i, :]).astype(float) / N_vecinos
            mat_copem[i, j + 1, :] = mat_copem[i, j, :] + columna

    # ------------------------------------------------------------
    # Acotar h para root 1
    # ------------------------------------------------------------
    h_clip = h.copy()
    min_m = np.min(muestra[:, :, 0], axis=1)
    max_m = np.max(muestra[:, :, 0], axis=1)

    h_clip[h_clip > max_m] = max_m[h_clip > max_m]
    h_clip[h_clip < min_m] = min_m[h_clip < min_m]

    # ------------------------------------------------------------
    # Soportes para inversión
    # ------------------------------------------------------------
    xx = Mat_xyord[:, :, 0]
    xm = np.concatenate(
        (xx[:, 0, np.newaxis],
         (xx[:, :-1] + xx[:, 1:]) / 2,
         xx[:, -1, np.newaxis]),
        axis=1
    )

    xv = Y_ord
    xmv = np.concatenate(
        (xv[:, 0, np.newaxis],
         (xv[:, :-1] + xv[:, 1:]) / 2,
         xv[:, -1, np.newaxis]),
        axis=1
    )

    k = np.arange(N_vecinos + 1)
    arange1 = np.arange(-1, N_vecinos)
    arange2 = np.arange(0, N_vecinos + 1)
    ajuste = np.concatenate(([-1], np.ones(N_vecinos)))

    # ------------------------------------------------------------
    # Root 1 (depende solo de h y de la muestra local)
    # ------------------------------------------------------------
    diff = np.zeros((n_puntos, N_vecinos + 1))

    for i, h0 in enumerate(h_clip):
        u_lower, u_middle, u_upper = 0.0, 0.5, 1.0

        f0 = np.sum(xm[i, :] * binom.pmf(k, N_vecinos, u_lower)) - h0
        if f0 == 0:
            u_middle = 0.0
        else:
            for _ in range(40):
                val_u_l = np.sum(xm[i, :] * binom.pmf(k, N_vecinos, u_lower)) - h0
                val_u_m = np.sum(xm[i, :] * binom.pmf(k, N_vecinos, u_middle)) - h0

                if val_u_l * val_u_m < 0:
                    u_upper = u_middle
                else:
                    u_lower = u_middle

                u_middle = (u_lower + u_upper) / 2.0

                if u_upper - u_lower < tolerancia:
                    break

        diff[i, :] = (
            binom.pmf(arange1, N_vecinos - 1, u_middle)
            - binom.pmf(arange2, N_vecinos - 1, u_middle) * ajuste
        )

    return {
        "N_vecinos": N_vecinos,
        "n_puntos": n_puntos,
        "mat_copem": mat_copem,
        "xmv": xmv,
        "diff": diff,
        "k": k,
        "arange2": arange2,
        "tolerancia": tolerancia
    }


def simular_desde_uu(precomp, uu_set):
    import numpy as np
    from scipy.stats import binom

    N_vecinos = precomp["N_vecinos"]
    n_puntos = precomp["n_puntos"]
    mat_copem = precomp["mat_copem"]
    xmv = precomp["xmv"]
    diff = precomp["diff"]
    k = precomp["k"]
    arange2 = precomp["arange2"]
    tolerancia = precomp["tolerancia"]

    m = uu_set.shape[0]
    sim_mtt = np.zeros((m, n_puntos))

    for i in range(n_puntos):
        diff_j = diff[i, :][:, np.newaxis]
        matriz_copem_j = mat_copem[i, :, :]
        xmv_j = xmv[i, :]

        for ii, u in enumerate(uu_set[:, i]):
            dbinom3 = binom.pmf(arange2, N_vecinos, 0.0)[:, np.newaxis]
            result = diff_j @ dbinom3.T

            if (N_vecinos * np.sum(matriz_copem_j * result) - u) == 0:
                v_middle = 0.0
            else:
                v_lower, v_middle, v_upper = 0.0, 0.5, 1.0

                for _ in range(40):
                    dbinom3 = binom.pmf(arange2, N_vecinos, v_lower)[:, np.newaxis]
                    result = diff_j @ dbinom3.T
                    val_v_l = N_vecinos * np.sum(matriz_copem_j * result) - u

                    dbinom3 = binom.pmf(arange2, N_vecinos, v_middle)[:, np.newaxis]
                    result = diff_j @ dbinom3.T
                    val_v_m = N_vecinos * np.sum(matriz_copem_j * result) - u

                    if val_v_l * val_v_m < 0:
                        v_upper = v_middle
                    else:
                        v_lower = v_middle

                    v_middle = (v_lower + v_upper) / 2.0

                    if v_upper - v_lower < tolerancia:
                        break

            sim_mtt[ii, i] = np.sum(xmv_j * binom.pmf(k, N_vecinos, v_middle))

    return sim_mtt


def Simula_Bernstein_cond(m, coords_Todos, N_vecinos, h_conocidos, K_conocidos, h, tree, tolerancia, uu_set=None, precomp=None):
    import numpy as np

    if precomp is None:
        precomp = preparar_Bernstein_cond(
            coords_Todos, N_vecinos, h_conocidos, K_conocidos, h, tree, tolerancia
        )

    if uu_set is None:
        uu_set = np.random.uniform(0, 1, (m, coords_Todos.shape[0]))
    else:
        m = uu_set.shape[0]

    return simular_desde_uu(precomp, uu_set)