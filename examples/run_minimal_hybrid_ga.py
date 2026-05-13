
"""
Minimal reproducible example for the Wasserstein--Bernstein Copula framework.

This script:
1. loads the benchmark dataset,
2. builds local Bernstein copula components,
3. optimizes one stochastic realization using a small two-stage hybrid GA,
4. compares the optimized realization with the mean and median of generated simulations.

The settings are intentionally small so the example can run quickly.
For the manuscript experiments, increase the population size and number of generations.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.stats import wasserstein_distance
from sklearn.metrics import mean_squared_error

# Allow imports from ../src when running this file from examples/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import calcular_vario_exp as cve
from Simula_Bernstein_optimizado import preparar_Bernstein_cond, Simula_Bernstein_cond


def binary_matrix_to_u(individual_bin: np.ndarray) -> np.ndarray:
    """Convert a binary chromosome of shape (num_var, nbits) into U(0,1) values."""
    nbits = individual_bin.shape[1]
    powers = 2 ** np.arange(nbits - 1, -1, -1)
    integers = individual_bin @ powers
    u = integers / (2**nbits - 1)
    return np.clip(u, 1e-6, 1 - 1e-6)


def u_to_binary(u: np.ndarray, nbits: int) -> np.ndarray:
    """Convert U(0,1) values into a binary chromosome."""
    u = np.clip(u, 0, 1)
    integers = np.rint(u * (2**nbits - 1)).astype(int)
    bits = ((integers[:, None] & (1 << np.arange(nbits - 1, -1, -1))) > 0).astype(int)
    return bits


def initial_population(size_population: int, num_var: int, nbits: int, rng: np.random.Generator) -> np.ndarray:
    """Random binary population with shape (size_population, num_var, nbits)."""
    return rng.integers(0, 2, size=(size_population, num_var, nbits), dtype=np.int8)


def hybrid_initial_population(
    best_individual: np.ndarray,
    size_population: int,
    nbits: int,
    rng: np.random.Generator,
    frac_random: float = 0.40,
    sigma_close: float = 0.05,
    sigma_medium: float = 0.10,
) -> np.ndarray:
    """Generate a hybrid population around the best solution from stage 1."""
    num_var = best_individual.shape[0]
    best_u = binary_matrix_to_u(best_individual)

    population = np.empty((size_population, num_var, nbits), dtype=np.int8)
    population[0] = best_individual.copy()

    n_random = max(1, int(size_population * frac_random))
    for p in range(1, size_population):
        if p <= n_random:
            u = rng.uniform(0, 1, size=num_var)
        elif p <= n_random + (size_population - n_random) // 2:
            u = best_u + rng.normal(0, sigma_close, size=num_var)
        else:
            u = best_u + rng.normal(0, sigma_medium, size=num_var)
        population[p] = u_to_binary(np.clip(u, 1e-6, 1 - 1e-6), nbits)

    return population


def compute_metrics(
    sim: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    known_mask: np.ndarray,
    K_known: np.ndarray,
    gamma_ref: np.ndarray,
    lag_value: float,
    N_lags: int,
    sigma_vario: float,
    sigma_data: float,
    weights: tuple[float, float, float],
) -> dict[str, float]:
    """Compute TOF components on known locations."""
    wv, wi, wd = weights

    xk = x[known_mask]
    yk = y[known_mask]
    sim_known = sim[known_mask]

    _, gamma_sim = cve.calcular_variograma_experimental(xk, yk, sim_known, lag_size=lag_value, N_lags=N_lags)

    # In case the last lag is empty in one of the samples, compare common lags only.
    L = min(len(gamma_ref), len(gamma_sim))
    vario_rmse = np.sqrt(mean_squared_error(gamma_ref[:L], gamma_sim[:L]))
    interp_rmse = np.sqrt(mean_squared_error(K_known, sim_known))
    wass = wasserstein_distance(K_known, sim_known)
    bounded_wass = 1 - np.exp(-wass)

    score = (wv / sigma_vario) * vario_rmse + (wi / sigma_data) * interp_rmse + wd * bounded_wass

    corr = np.corrcoef(K_known, sim_known)[0, 1]

    return {
        "score": float(score),
        "variogram_error": float((1 / sigma_vario) * vario_rmse),
        "interpolation_error": float((1 / sigma_data) * interp_rmse),
        "correlation": float(corr),
        "wasserstein": float(wass),
        "bounded_wasserstein": float(bounded_wass),
    }


def evaluate_individual(individual_bin: np.ndarray, precomp: dict, metric_args: dict) -> tuple[float, np.ndarray, dict]:
    """Generate one complete realization and evaluate it."""
    u = binary_matrix_to_u(individual_bin)
    sim = Simula_Bernstein_cond(
        m=1,
        coords_Todos=metric_args["coords_all"],
        N_vecinos=metric_args["N_vecinos"],
        h_conocidos=metric_args["h_known"],
        K_conocidos=metric_args["K_known"],
        h=metric_args["h_all"],
        tree=metric_args["tree"],
        tolerancia=metric_args["tolerancia"],
        uu_set=u.reshape(1, -1),
        precomp=precomp,
    )[0]

    metrics = compute_metrics(sim=sim, **metric_args["metrics"])
    return metrics["score"], sim, metrics


def tournament_select(scores: np.ndarray, tournament_size: int, rng: np.random.Generator) -> int:
    """Return the index of the best individual among a random tournament."""
    candidates = rng.choice(len(scores), size=tournament_size, replace=False)
    return candidates[np.argmin(scores[candidates])]


def evolve_population(
    population: np.ndarray,
    scores: np.ndarray,
    elite_size: int,
    tournament_size: int,
    prob_cross: float,
    prob_mut: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply elitism, tournament selection, one-point crossover, and bit mutation."""
    size_population, num_var, nbits = population.shape
    new_population = np.empty_like(population)

    elite_idx = np.argsort(scores)[:elite_size]
    new_population[:elite_size] = population[elite_idx]

    for p in range(elite_size, size_population):
        p1 = population[tournament_select(scores, tournament_size, rng)]
        p2 = population[tournament_select(scores, tournament_size, rng)]

        child = p1.copy()
        if rng.random() < prob_cross:
            cut = rng.integers(1, num_var)
            child[cut:] = p2[cut:]

        mutation_mask = rng.random(size=child.shape) < prob_mut
        child[mutation_mask] = 1 - child[mutation_mask]

        new_population[p] = child

    return new_population


def run_stage(stage_name: str, population: np.ndarray, n_generations: int, precomp: dict, metric_args: dict, ga_params: dict, rng: np.random.Generator):
    """Run one GA stage."""
    best = {"score": np.inf, "individual": None, "sim": None, "metrics": None, "stage": stage_name, "generation": None}
    all_sims = []

    for gen in range(n_generations):
        scores = np.zeros(len(population))

        for i, individual in enumerate(population):
            score, sim, metrics = evaluate_individual(individual, precomp, metric_args)
            scores[i] = score
            all_sims.append(sim)

            if score < best["score"]:
                best.update({
                    "score": score,
                    "individual": individual.copy(),
                    "sim": sim.copy(),
                    "metrics": metrics,
                    "generation": gen,
                })

        print(f"{stage_name} | generation {gen + 1:03d}/{n_generations} | best score: {best['score']:.6f}")

        population = evolve_population(
            population,
            scores,
            elite_size=ga_params["elite_size"],
            tournament_size=ga_params["tournament_size"],
            prob_cross=ga_params["prob_cross"],
            prob_mut=ga_params["prob_mut"],
            rng=rng,
        )

    return population, best, all_sims


def main() -> None:
    rng = np.random.default_rng(123)

    # ---------------------------
    # User-adjustable parameters
    # ---------------------------
    N_vecinos = 6
    tolerancia = 1e-4
    lag_value = 236
    N_lags = 15
    weights = (1 / 3, 1 / 3, 1 / 3)

    # Small GA settings for a fast demonstration
    size_population = 4
    nbits = 6
    generations_stage1 = 2
    generations_stage2 = 2

    ga_params = {
        "elite_size": 2,
        "tournament_size": 3,
        "prob_cross": 0.6,
        "prob_mut": 1 / (nbits * 80),
    }

    # ---------------------------
    # Load data
    # ---------------------------
    data_path = PROJECT_ROOT / "data" / "Datos_hess_400_datos.csv"
    df_full = pd.read_csv(data_path)

    # Use a small subset for a fast reproducible example.
    # The full manuscript experiments use the complete dataset.
    max_known = 40
    max_unknown = 40
    df_known = df_full[df_full["contadores"] == 1].head(max_known)
    df_unknown = df_full[df_full["contadores"] == 0].head(max_unknown)
    df = pd.concat([df_known, df_unknown], ignore_index=True)

    x = df["x_utm"].to_numpy()
    y = df["y_utm"].to_numpy()
    h = df["h"].to_numpy()
    K = df["lnk"].to_numpy()
    known_mask = df["contadores"].to_numpy() == 1

    coords_all = np.column_stack([x, y])
    coords_known = coords_all[known_mask]

    h_known = h[known_mask]
    K_known = K[known_mask]

    tree = KDTree(coords_known)

    # Reference variogram from known hydraulic conductivity data
    _, gamma_ref = cve.calcular_variograma_experimental(
        x[known_mask], y[known_mask], K_known, lag_size=lag_value, N_lags=N_lags
    )

    sigma_vario = np.std(gamma_ref, ddof=1)
    sigma_data = np.std(K_known, ddof=1)

    print("Preparing local Bernstein copula components...")
    precomp = preparar_Bernstein_cond(
        coords_Todos=coords_all,
        N_vecinos=N_vecinos,
        h_conocidos=h_known,
        K_conocidos=K_known,
        h=h,
        tree=tree,
        tolerancia=tolerancia,
    )

    metric_args = {
        "coords_all": coords_all,
        "N_vecinos": N_vecinos,
        "h_known": h_known,
        "K_known": K_known,
        "h_all": h,
        "tree": tree,
        "tolerancia": tolerancia,
        "metrics": {
            "x": x,
            "y": y,
            "known_mask": known_mask,
            "K_known": K_known,
            "gamma_ref": gamma_ref,
            "lag_value": lag_value,
            "N_lags": N_lags,
            "sigma_vario": sigma_vario,
            "sigma_data": sigma_data,
            "weights": weights,
        },
    }

    num_var = len(coords_all)

    # ---------------------------
    # Stage 1: exploratory GA
    # ---------------------------
    population_stage1 = initial_population(size_population, num_var, nbits, rng)
    _, best_stage1, sims1 = run_stage(
        "Stage 1 exploratory",
        population_stage1,
        generations_stage1,
        precomp,
        metric_args,
        ga_params,
        rng,
    )

    # ---------------------------
    # Stage 2: hybrid refinement
    # ---------------------------
    population_stage2 = hybrid_initial_population(best_stage1["individual"], size_population, nbits, rng)
    _, best_stage2, sims2 = run_stage(
        "Stage 2 hybrid",
        population_stage2,
        generations_stage2,
        precomp,
        metric_args,
        ga_params,
        rng,
    )

    best = best_stage1 if best_stage1["score"] <= best_stage2["score"] else best_stage2

    # Baseline: mean and median of all GA-generated simulations
    all_sims = np.vstack(sims1 + sims2)
    sim_mean = np.mean(all_sims, axis=0)
    sim_median = np.median(all_sims, axis=0)

    mean_metrics = compute_metrics(sim_mean, **metric_args["metrics"])
    median_metrics = compute_metrics(sim_median, **metric_args["metrics"])

    summary = pd.DataFrame([
        {"method": "optimized_TOF", **best["metrics"]},
        {"method": "BCSCS_mean_like", **mean_metrics},
        {"method": "BCSCS_median_like", **median_metrics},
    ])

    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(exist_ok=True)

    summary.to_csv(output_dir / "example_summary.csv", index=False)

    final = df.copy()
    final["K_sim_optimized_TOF"] = best["sim"]
    final["K_sim_mean"] = sim_mean
    final["K_sim_median"] = sim_median
    final.to_csv(output_dir / "example_simulated_fields.csv", index=False)

    print("\nBest solution:")
    print(f"Stage: {best['stage']}")
    print(f"Generation: {best['generation']}")
    print(summary.round(6).to_string(index=False))

    print("\nFiles written:")
    print(output_dir / "example_summary.csv")
    print(output_dir / "example_simulated_fields.csv")


if __name__ == "__main__":
    main()
