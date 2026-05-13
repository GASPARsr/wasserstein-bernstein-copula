# Wasserstein-Bernstein Copula Spatial Simulation Framework

Python implementation of a stochastic spatial simulation framework for modeling hydraulic conductivity in heterogeneous aquifers using local Bernstein copulas, Wasserstein distance, and a two-stage hybrid genetic algorithm.

## Overview

This repository contains the implementation associated with the manuscript:

**A Wasserstein Distance and Bernstein Copula Framework for Modeling the Spatial Variability of Hydraulic Conductivity**

The framework combines:

- Local nonparametric Bernstein copulas
- Tri-objective optimization
- Experimental variogram consistency
- Wasserstein-based distribution preservation
- Two-stage hybrid genetic optimization

---

## Repository structure

```text
src/        Core implementation
examples/   Reproducible benchmark example
data/       Example benchmark dataset
```

---

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Reproducible example

Run the benchmark example with:

```bash
python examples/run_minimal_example.py
```

Expected output:

```text
Simulation completed successfully
Interpolation error: [value]
Wasserstein distance: [value]
```

---

## Benchmark dataset

The included example dataset is based on the synthetic groundwater benchmark scenario used for methodological validation.

---

## Reproducibility

For reproducibility, the example uses a fixed random seed.

---

## License

MIT License

---

## Citation

If you use this code, please cite:

Salas-Ruelas, G. et al. (2026)

A Wasserstein Distance and Bernstein Copula Framework for Modeling the Spatial Variability of Hydraulic Conductivity
