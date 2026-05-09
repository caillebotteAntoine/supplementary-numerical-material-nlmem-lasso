# Supplementary numerical material for the NLMEM-LASSO procedure

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Poetry](https://img.shields.io/badge/dependency%20manager-poetry-60A5FA.svg)](https://python-poetry.org/)

Supplementary material for the paper "Estimation and variable selection in high dimension in nonlinear mixed-effects models."

This repository contains:
- real datasets and script for data simulation,
- model definitions for LMEM, NLMEM, and PKMEM settings,
- scripts to reproduce Figures 1 to 9 from the paper,
- post-processing utilities for selection results and SNP visualization.

This repository do not contain the main `sdg4varselect` package code, which contain the NLMEM-LASSO procedure and is available separately at https://github.com/caillebotteAntoine/sdg4varselect. The associated documentation for the package is available at [documentation site](https://sdg4varselect.readthedocs.io/en/latest/)

## Project layout

```text
.
├── figure_1.py ... figure_9.py      # Figure scripts used in the manuscript
├── application/                     # Real-data pipeline and plotting helpers
│   ├── data/                        # Input data (senescence, SNP maps, chromosome files)
│   └── results/                     # Selected SNP outputs
├── simulation/                      # Simulation workflows and cluster scripts
│   ├── models/                      # LMEM/NLMEM/PKMEM model definitions
│   └── results/                     # Simulation outputs used by figure scripts
├── utils/                           # Shared utilities for loading and plotting selection scores
└── pyproject.toml                   # Project metadata and dependencies
```

## Methods overview

The codebase covers three methodological settings:
- LMEM-LASSO workflows for high-dimensional linear mixed-effects models.
- NLMEM-LASSO workflows for nonlinear logistic mixed-effects models.
- PKMEM simulations and two-step comparisons for pharmacological scenarios.

Regularization-path and support-selection summaries are used to report underselection, exact support recovery, and overselection.

## Data

- Real data for the application is stored in `application/data/`.
- Simulation-derived result files used by the plotting scripts are stored under `simulation/results/` and `application/results/`.
- Figure scripts typically read precomputed CSV/result artifacts and produce publication-ready visualizations.

## Getting started

### 1. Clone and enter the repository

```bash
git clone AFAIRE
cd supplementary-material-nlmem-lasso
```

### 2. Create environment and install dependencies

Using Poetry (recommended):

```bash
poetry install
```

## Usage examples

### Reproduce figure scripts

Run any figure script from the repository root:

```bash
python figure_2.py
python figure_4.py
python figure_7.py
```

Notes:
- Several scripts create figure objects but keep `savefig(...)` calls commented.
- Uncomment the corresponding `savefig` lines in each script if you want files written to disk.

### Real-data application workflow snippets

```bash
python figure_8.py
python figure_9.py
```

`figure_8.py` loads senescence data for a chromosome and plots selected genotype trajectories.
`figure_9.py` aggregates selected SNPs across chromosomes and prepares combined outputs.

### Simulation scripts

The `simulation/` folder contains cluster-oriented scripts (`lmem_cluster.py`, `nlmem_cluster.py`, `pkmem_cluster.py`) and model definitions under `simulation/models/`.

## Reproducibility

- Use a fixed Python version compatible with `pyproject.toml` (3.10+).
- Run scripts from repository root so relative data/result paths resolve correctly.
- Keep input data and results files in their current folder structure; many scripts use hard-coded relative paths.

## Where to get help

- Check the documentation of the sdg4varselect package for details on the underlying methods.
- Open an issue in the repository for reproducibility or bug reports.
- Contact the maintainer: `caillebotte.antoine@gmail.com`.
