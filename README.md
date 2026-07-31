# Code Repo for _MoRe: Modular Representations for Principled Continual Representation Learning on Sequential Data_

*The updated gating mechanism has been updated in `synthetic/synthetic_adaptation`, and the original version is saved at `synthetic/synthetic_adaptation_archived` for comparison.*

## Overview

There are two directories corresponding to synthetic studies and real-world LLM activation evaluations, namely, `synthetic/` and `realworld/`.

```
.
├── synthetic/
│   ├── identifiability/                    # S-RQ1
│   ├── adaptation/                         # S-RQ2 and S-RQ3
│   └── synthetic_estimate_single_layer/    # [Optional] Layer-wise estimation
└── realworld/
    ├── interpretability/                   # RW-RQ1
    └── downstream/                         # RW-RQ2
```

## Synthetic Studies

### S-RQ1 Identifiability

`synthetic/identifiability/`

### S-RQ2 Plasticity and Stability

Code layout under `synthetic/adaptation/`

| File                  | Purpose                                                                    |
| --------------------- | -------------------------------------------------------------------------- |
| `more_core.py`        | MoRe model, training loop, data generation, evaluation utilities (shared). |
| `more_baselines.py`   | Runs all reported baselines and MoRe variants for S-RQ2 and S-RQ3.         |
| `more_sweep.py`       | Hyperparameter sweep for the four MoRe variants across the 6 cases.        |

This experiment evaluates plasticity (target MCC) and stability (forgetting) across six transition scenarios (C1–C6 in the paper; see `ALL_TRANSITIONS` in `more_core.py`, indices 0–5). The seventh case in the data-generation script (`case7_z2_only`) is excluded from reported results.

Baselines and ablation reported in Fig. 3a:

| Method            | Description                                                  |
| ----------------- | ------------------------------------------------------------ |
| `scratch`         | Train a new encoder on data2 from random init (no transfer) |
| `finetune`        | Initialize from data1 encoder, fine-tune all params on data2 |
| `expansion_only`  | Freeze data1 encoder, learn new modules only                 |
| `MoRe_no_T`       | Full MoRe but without alignment matrix `T` (ablation)        |
| `MoRe`            | Full proposed method                                         |

**Run.**

```bash
cd synthetic/adaptation
# Full reproduction (3 seeds, 6 cases, T=50k, 5k epochs):
python more_baselines.py \
    --out_dir results/baselines \
    --T 50000 --epochs 5000 \
    --seeds 0 1 2

```

Outputs:

- `results/baselines/metrics.csv` — one row per `(seed, transition, method)`
- `results/baselines/summary.json` — mean/std of `target_mcc`, `old_after_mcc`, `forgetting`, `gate_acc` per `(transition, method)`
- `results/baselines/seed<s>_<transition>_sc_diag.pt` — saved analytic self-consistency scores (diagnostic)

### S-RQ3 Gate selection accuracy

`synthetic/adaptation/`

This experiment evaluates whether reused and expanded modules are correctly assigned across the same six transitions. Methods reported in Fig. 3b:

| Method            | Description                                          |
| ----------------- | ---------------------------------------------------- |
| `MoRe`            | Full method (both sparsity penalties on)             |
| `MoRe_no_sp_old`  | MoRe without L0 sparsity on old gates (`w_sp_old=0`) |
| `MoRe_no_sp_new`  | MoRe without L0 sparsity on new gates (`w_sp_new=0`) |

S-RQ3 reuses the same run as S-RQ2: `more_baselines.py` produces all three MoRe variants in a single execution (along with `MoRe_no_T` for S-RQ2). The `gate_acc` column in `metrics.csv` provides the metric reported in Fig. 3b. No separate command is needed beyond the S-RQ2 command above.

### Hyperparameter sweep (used to select the reported configuration)

`synthetic/adaptation/more_sweep.py`

This sweeps over the loss weights and temperature parameters of the four MoRe variants (`MoRe_no_T`, `MoRe`, `MoRe_no_sp_old`, `MoRe_no_sp_new`) across the six reported cases. It is the procedure used to select the per-scenario hyperparameters described in Appendix E.4.

```bash
cd synthetic/adaptation

python more_sweep.py \
    --out_dir results/sweep \
    --seeds 0 1 2 \
    --max_configs 30

```

Outputs: `results/sweep/metrics_seeds<seeds>_cases<cases>.csv` — one row per `(seed, hp_config, transition, method)`. Filenames are namespaced by seeds and cases to support running multiple jobs in parallel without collisions.

### [Optional] Layer-wise estimation

`synthetic/synthetic_estimate_single_layer/`

## Real-world Experiments

The real-world experiments train MoRe on token-level activations extracted from open-source LLMs (Pythia-1.4B at layer 16; optionally Gemma-2-9B at layer 25), and evaluate (a) whether the learned hierarchy aligns with externally defined abstraction levels (RW-RQ1) and (b) whether the learned representations are label-efficient for downstream classification (RW-RQ2).

Datasets and label sources are identical across both experiments:

| Dataset     | Source                                  | Label proxy                                                                |
| ----------- | --------------------------------------- | -------------------------------------------------------------------------- |
| arXiv       | local JSONL (`arxiv_metadata.json`)     | `categories` field → 4-tier mapping in `ARXIV_TIER`                        |
| AMPS        | HuggingFace `sarahpann/AMPS`            | keyword matching against problem text (arithmetic → advanced linear algebra) |
| Wikipedia   | HuggingFace `wikimedia/wikipedia` | article word count + title keywords (general → specific)               |

### RW-RQ1 Modular representation verification

`realworld/interpretability/`

This experiment trains a linear hierarchical VAR-style MoRe (4 layers × 64 concepts) on LLM activations, then evaluates (i) per-layer concept concentration relative to externally defined abstraction tiers and (ii) cross-layer dependencies via the learned `B`-matrices (Tables 2 and 5, Figure 4 in the paper).

The pipeline runs in two stages:

**Stage 1 — Extract activations, fit PCA, train MoRe.**

`run_pipeline.py` performs activation extraction (with hooks at the configured LLM layer, streaming token rows directly to disk for O(1) RAM), PCA compression to 256 dims, and training of the hierarchical VAR-style MoRe across a grid of model orders `p ∈ {10, 50, 100}` and inter-layer sparsity weights `λ_sp ∈ {1e-4, 1e-5}`. 

```bash
cd realworld/interpretability

# Stage 1a: extract activations once for each (model, dataset) 
python run_pipeline.py --stage extract --output_dir ./hier_sae_outputs

# Stage 1b: train MoRe 
python run_pipeline.py --stage train --output_dir ./hier_sae_outputs
```

Paths inside `run_pipeline.py` (LLM names, layer indices, the local arXiv JSONL path, output directory) are configured in the `Config` dataclass at the top of the file — edit these to match your cluster before running.

**Stage 2 — Interpretation: concentration and cross-layer links.**

`interpre.py` loads the trained checkpoints for all `(model_order, λ_sp)` combinations and the corresponding LLM activations, then produces the two diagnostics reported in the paper:

1. Concept concentration — CV-normalized concentration coefficient per concept, with a null control obtained by shuffling tier labels (Figure 4; Table 2).
2. Cross-layer `B`-matrix links — top-`k` directed concept-to-concept links, with TF-IDF labels of the top-activating passages (Table 5).

```bash
cd realworld/interpretability

# Edit OUTPUT_DIR at the top of interpre.py to point at the directory produced by run_pipeline.py.
python interpret.py
```

Outputs are written under `<output_dir>/interpretation/`:

- `gini_<combo>.pdf` — per-combo concept-concentration violin plots with null control overlay
- `crosslinks_<combo>.pdf` — top L0→L3 cross-level links as a bipartite graph
- `block_strengths_<combo>.json` — full mean/max weight of every L_src → L_tgt cross-block, with top-3 link details
- `summary_gini.csv` — per `(dataset, model_order, λ_sp, layer)` concentration mean/std and null mean

### RW-RQ2 Classification efficiency by modular representation

`realworld/downstream/`

This experiment evaluates label efficiency by training a linear probe on frozen MoRe representations to predict the same tier labels used in RW-RQ1, varying the number of labeled passages from 10 to 1,000 (Table 3 in the paper). The probe compares raw PCA features against per-layer MoRe representations (`MoRe_L0` … `MoRe_L3`) and the full concatenated MoRe representation (`MoRe_all`).

<!-- The labeling logic is ported verbatim from `interpret.py` so RW-RQ1 and RW-RQ2 use exactly the same labels (no `.categories.npy` dependency; labels are recomputed from the original text sources). -->

**Run.** RW-RQ2 depends on the outputs of `run_pipeline.py` (activations, PCA, and trained checkpoints). 
```bash
cd realworld/downstream

python rw_rq2_probe.py \
    --output_dir /path/to/hier_sae_outputs \
    --results_dir results/rw_rq2 \
    --models pythia1b4 gemma2_9b \
    --datasets amps arxiv wikipedia \
    --arxiv_local_path /path/to/arxiv_metadata.json \
    --model_order 10 --lambda_sp 1e-4 --epoch 30 \
    --n_layers 4 --z_per_layer 64 \
    --n_seeds 5 --min_freq 50
```

Before launching `sbatch_rw_rq2.sh`, edit the `OUTPUT_DIR` variable inside the script to point at the directory produced by `run_pipeline.py`. The whole probe runs in roughly 30 minutes on a single GPU (sklearn linear probes; GPU is used only for representation construction).


Outputs are written under `--results_dir`:

- `probe_results.csv` — one row per `(model, dataset, representation, n_labeled)` with mean/std accuracy across seeds
- `probe_results.json` — same data in nested form
- `figures/probe_<model>__<dataset>.pdf` — label-efficiency curves (accuracy vs. number of labeled passages), one panel per `(model, dataset)`, with all 6 representations overlaid

---



## Code layout under `realworld/`

| File                                          | Purpose                                                                |
| --------------------------------------------- | ---------------------------------------------------------------------- |
| `realworld/interpretability/run_pipeline.py` | Extract LLM activations, fit PCA, train hierarchical MoRe (SLURM array). |
| `realworld/interpretability/interpret.py`     | Concept concentration, and cross-layer `B`-matrix links.     |
| `realworld/downstream/rw_rq2_probe.py`       | Linear-probe label efficiency on frozen MoRe vs. PCA representations.  |
| `realworld/downstream/sbatch_rw_rq2.sh`      | SLURM submission script wrapping `rw_rq2_probe.py`.                    |

## Environment

```
python >= 3.9
torch >= 2.0
numpy
scikit-learn       # PCA, linear probe
transformers       # LLM loading for activation extraction
datasets           # HuggingFace streaming for AMPS / Wikipedia
pandas, matplotlib, seaborn   # interpretation figures
```

A single A6000 (or comparable 48 GB GPU) is sufficient for all experiments. Activation extraction for the largest dataset (arXiv, 50k passages, Pythia-1.4B) consumes ~20 GB of disk (while for Gemma-2-9B it needs susbtantially more disk); the script streams token rows directly to disk so peak RAM stays at the size of one extraction batch (~50 MB).