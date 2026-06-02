# Modular Representation for Continual Learning

## Scope

This repository folder contains the synthetic experiment code used for the current MoRe representation-learning experiments.

The main synthetic code path is:

- dataset generation from configurable hierarchical temporal processes
- training the first-layer estimator on those datasets
- saving checkpoints, metrics, and plots for the synthetic runs

The current work does **not** package the variance-sweep study as part of the main reproducibility path.

## Main Synthetic Experiment Code

The main runner for the general synthetic experiments is:

- [synthetic_estimate_single_layer/run_first_layer_stability.py](../synthetic_estimate_single_layer/run_first_layer_stability.py)

This depends on:

- [synthetic_estimate_single_layer/first_layer_stability.py](../synthetic_estimate_single_layer/first_layer_stability.py)
- [synthetic_estimate_single_layer/data/temporal_generator.py](../synthetic_estimate_single_layer/data/temporal_generator.py)
- [synthetic_estimate_single_layer/data/temporal_cfg.py](../synthetic_estimate_single_layer/data/temporal_cfg.py)
- [synthetic_estimate_single_layer/data/run_generator_multiple.sh](../synthetic_estimate_single_layer/data/run_generator_multiple.sh)

These files are sufficient to:

1. generate the synthetic datasets
2. run the synthetic training experiments
3. save the resulting plots, metrics, and checkpoints

## Minimal Upload Set for General Synthetic Experiments

If you want to upload the general synthetic experiment code, upload at least:

- [synthetic_estimate_single_layer/first_layer_stability.py](../synthetic_estimate_single_layer/first_layer_stability.py)
- [synthetic_estimate_single_layer/run_first_layer_stability.py](../synthetic_estimate_single_layer/run_first_layer_stability.py)
- [synthetic_estimate_single_layer/data/temporal_generator.py](../synthetic_estimate_single_layer/data/temporal_generator.py)
- [synthetic_estimate_single_layer/data/temporal_cfg.py](../synthetic_estimate_single_layer/data/temporal_cfg.py)
- [synthetic_estimate_single_layer/data/run_generator_multiple.sh](../synthetic_estimate_single_layer/data/run_generator_multiple.sh)

Recommended to include as well:

- [synthetic_estimate_single_layer/estimate_first_layer.ipynb](../synthetic_estimate_single_layer/estimate_first_layer.ipynb)
- [synthetic_estimate_single_layer/estimate_first_layer.py](../synthetic_estimate_single_layer/estimate_first_layer.py)

## Paper Figure Reproduction

The four synthetic normal-training figures used in the paper appendix have a dedicated reproduction path:

- [synthetic_estimate_single_layer/run_paper_synthetic_figures.py](../synthetic_estimate_single_layer/run_paper_synthetic_figures.py)
- [synthetic_estimate_single_layer/make_paper_synthetic_figures.py](../synthetic_estimate_single_layer/make_paper_synthetic_figures.py)
- [synthetic_estimate_single_layer/PAPER_FIGURE_REPRODUCIBILITY.md](../synthetic_estimate_single_layer/PAPER_FIGURE_REPRODUCIBILITY.md)

That path is only for reproducing the four appendix figures and is separate from the general synthetic experiment runner above.
