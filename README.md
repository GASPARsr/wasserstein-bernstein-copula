# Wasserstein--Bernstein Copula Framework

This repository contains a minimal reproducible Python example for stochastic spatial simulation of hydraulic conductivity using:

- local Bernstein copulas,
- a tri-objective function,
- Wasserstein distance,
- and a two-stage hybrid genetic algorithm.

The example is intentionally small so that it can be executed quickly and used as a starting point for reproducing the methodology described in the manuscript. It uses a reduced subset of the benchmark data for demonstration. The full manuscript experiments use the complete dataset and larger genetic algorithm settings.

## Repository structure

```text
wasserstein-bernstein-copula/
├── data/
│   └── Datos_hess_400_datos.csv
├── examples/
│   └── run_minimal_hybrid_ga.py
├── results/
├── src/
│   ├── calcular_vario_exp.py
│   ├── FTO.py
│   └── Simula_Bernstein_optimizado.py
├── LICENSE
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation

Create a Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

## Running the example

From the repository root, run:

```bash
python examples/run_minimal_hybrid_ga.py
```

The script generates two output files in the `results/` folder:

```text
example_summary.csv
example_simulated_fields.csv
```

## Notes

The genetic algorithm settings in the example are deliberately small:

- population size: 8
- Stage 1 generations: 4
- Stage 2 generations: 4

These settings are designed for demonstration only. For research experiments, increase the population size, number of generations, and number of bits according to the manuscript configuration.

## Citation

If you use this code, please cite the associated manuscript:

Salas-Ruelas et al. (2026). A Wasserstein Distance and Bernstein Copula Framework for Modeling the Spatial Variability of Hydraulic Conductivity.
