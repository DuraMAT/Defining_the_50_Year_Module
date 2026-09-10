# 50-Year PV Module Reliability and Hail Simulation

This directory contains the analysis code and input data used to evaluate long-term PV module survivability under intrinsic reliability failures and stochastic hail exposure, presented in the paper "Defining the 50 Year Module"

The model combines:

- Weibull intrinsic module reliability
- Location-specific hail occurrence and return periods
- Spatial hail-coverage corrections
- Location-specific hail-size distributions
- Conditional broken-glass damage distributions
- Post-hail performance loss rate (PLR)
- Monte Carlo simulation of physical and performance-adjusted capacity over a 50-year lifetime

## Files

- `hail_model.py` — Functions for hail occurrence, damage sampling, Weibull reliability, post-hail degradation, and Monte Carlo project simulation.
- `TBD.ipynb` — Main analysis notebook used to load model inputs, run individual and Monte Carlo simulations, compare locations, and generate manuscript figures.
- Hail input files — Location-specific return periods, hail-size distributions, spatial exposure information, and empirical damage/PLR distributions used by the model.

## Model outputs

The simulations distinguish between:

- **Physical capacity** — capacity remaining after intrinsic reliability and catastrophic hail failures.
- **Effective capacity** — physical capacity adjusted for additional performance degradation of hail-exposed surviving modules.

Simulations are evaluated over a 50-year module lifetime, with Year 30 used as a primary project-level comparison point in the current analysis.

## Status

Research code under active development. Model assumptions, input-data sources, and analysis methods are documented in the associated manuscript and notebook.