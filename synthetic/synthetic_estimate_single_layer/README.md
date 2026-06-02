# Modular Representation Synthetic Experiments

This directory contains the synthetic experiment code used for the current MoRe first-layer representation-learning studies.

The code supports:

1. generating synthetic hierarchical temporal datasets
2. training the first-layer estimator on those datasets
3. saving checkpoints, metrics, reports, and plots
4. reproducing the four synthetic figures used in the paper appendix

## Repository Layout

Main experiment code:

- `synthetic_estimate_single_layer/first_layer_stability.py`
- `synthetic_estimate_single_layer/run_first_layer_stability.py`

Synthetic data generation:

- `../data/temporal_generator.py`
- `../data/temporal_cfg.py`
- `../data/run_generator_multiple.sh`

Paper-figure reproduction:

- `synthetic_estimate_single_layer/run_paper_synthetic_figures.py`
- `synthetic_estimate_single_layer/make_paper_synthetic_figures.py`
- `synthetic_estimate_single_layer/PAPER_FIGURE_REPRODUCIBILITY.md`

Optional notebook:

- `synthetic_estimate_single_layer/estimate_first_layer.ipynb`

## Environment Setup

Example setup with conda:

```bash
conda create -n more_synth python=3.11 -y
conda activate more_synth
pip install numpy pandas matplotlib nbformat jupyter
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Notes:

- A CUDA-enabled PyTorch installation is recommended for the full training runs.
- The CPU build is sufficient for correctness but can be slow.

## How to Reproduce the Main Synthetic Experiments

From this directory:

```bash
cd synthetic_estimate_single_layer
python run_first_layer_stability.py
```

This script will:

1. generate the required synthetic datasets from the presets in `data/temporal_cfg.py`
2. train the synthetic experiments
3. save metrics tables, plots, checkpoints, and a report notebook

The outputs are written to a timestamped folder under:

```text
synthetic_estimate_single_layer/experiments/first_layer_stability_<timestamp>/
```

Useful command variants:

Force regeneration of datasets:

```bash
python run_first_layer_stability.py --force-generate
```

Full run with explicit output directory:

```bash
python run_first_layer_stability.py \
  --experiment-root experiments/first_layer_stability_full \
  --n-epochs 1000 \
  --n-q-epochs 20 \
  --n-f-epochs 3 \
  --warmup-epochs 50
```


## Generated Outputs

For each run, the experiment directory contains artifacts such as:

- generated datasets copied into `datasets/`
- per-run metrics CSV files
- scatter plots and training curves
- model checkpoints
- summary tables
- a rendered HTML and notebook report
