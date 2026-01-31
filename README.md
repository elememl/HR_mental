# Mental Health in Tech (OSMI 2016) — Clean Client Repo

This folder is a minimal, self-contained client delivery for running the unified HR segmentation analysis.

## What’s included

- `analysis.ipynb`: the full analysis notebook (run this).
- `data/mental-heath-in-tech-2016_20161114.csv`: the dataset used by the notebook.
- `environment.yaml`: conda environment specification (recommended).
- `requirements.txt`: pip requirements (fallback).

## Run locally

From this directory:

1) Create/activate environment

- Conda (recommended): `conda env create -f environment.yaml && conda activate mental`
- Pip (fallback): `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`

2) Launch notebook

- `jupyter notebook analysis.ipynb`

## Notes

UMAP is used primarily for visualization and for demonstrating how nonlinear embeddings can change apparent separability. Cluster interpretation emphasizes stability and simple, HR-actionable levers; overlays on mental-health outcomes are descriptive (association, not causation).
