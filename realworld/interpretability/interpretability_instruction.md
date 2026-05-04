# Hierarchical SAE — Collaborator Guide

## Goal

Train a linear VAR generative model on LLM activations to **recover a concept hierarchy unsupervised**, then verify it against dataset labels. Three datasets, two LLMs, swept over model order and sparsity penalty.

---

## Files

```
run_pipeline.py       # full pipeline: extraction → PCA → training
submit_hier_sae.sh    # SLURM array job script (24 jobs)
```

---

## Step 0 — One-time data download

The arXiv dataset must be downloaded once before running anything:

```bash
wget -c "https://huggingface.co/datasets/jackkuo/arXiv-metadata-oai-snapshot/resolve/main/arxiv-metadata-oai-snapshot.json" \
     -O /data/user_data/jiaqisun/arxiv_metadata.json
```

~3.5 GB, ~10 min. The `-c` flag resumes if interrupted.

---

## Step 1 — Environment

```bash
conda activate idol
pip install transformers datasets scikit-learn numpy torch
```

Set HuggingFace cache if needed:
```bash
export HF_HOME=/your/shared/cache/hf
```

---

## Step 2 — Run

```bash
# edit OUTPUT_DIR in submit_hier_sae.sh if needed, then:
sbatch submit_hier_sae.sh
```

Task 0 runs extraction for all datasets/models first (writes a lockfile), then all 24 tasks train in parallel. Each task is one `(model, dataset, model_order, lambda_sp)` combination.

---

## Datasets

Three datasets, each trained **independently**. Labels are used only for post-hoc evaluation — not during training.

| Dataset | Source | Hierarchy defined by | Label field |
|---|---|---|---|
| `arxiv` | local JSONL (`arxiv_metadata.json`) | arXiv subject category | `categories` e.g. `"math.LO cs.LG"` |
| `amps` | `sarahpann/AMPS` (HuggingFace) | Math topic module + difficulty | `problem` + `step_by_step` fields |
| `wikipedia` | `wikimedia/wikipedia 20231101.en` | Wikipedia category depth | article `title` (proxy) |

**Expected hierarchy per dataset:**

```
arxiv    : math.LO → math.CA → cs.LG → q-bio / econ  (fundamental → specific)
amps     : arithmetic → algebra → calculus → eigenvalues / KL-divergence
wikipedia: Mathematics → Algebra → Linear algebra → Black-Scholes model
```

The model sees none of these labels during training. After training, labels are used to check whether Layer 0 concepts activate broadly (math.LO papers) and Layer 3 concepts activate narrowly (q-bio papers).

---

## Model sweep (24 SLURM jobs)

```
2 LLMs  ×  3 datasets  ×  model_order ∈ {10, 50, 100}  ×  lambda_sp ∈ {1e-4, 1e-5}
```

| | |
|---|---|
| **Pythia-1.4B** | extract from layer 16, d_model=2048 |
| **Llama-3-8B** | extract from layer 20, d_model=4096 |
| **PCA** | 256 components, fit on AMPS, applied to all datasets |
| **model_order** | number of lag matrices B1..Bp in the VAR |
| **lambda_sp** | sparsity on inter-layer B blocks (smaller = looser hierarchy) |

---

## Output structure

```
hier_sae_outputs/
  activations/
    pythia1b4__arxiv.npy              # float16 [N_tokens, d_model]
    pythia1b4__arxiv.boundaries.npy   # int64 [N_passages+1] passage start indices
    pythia1b4__arxiv.categories.npy   # object [N_passages] category strings
    pythia1b4__amps.npy
    ...
  pca/
    pythia1b4__pca.npz                # components, mean, explained_variance
    llama3_8b__pca.npz
  checkpoints/
    pythia1b4__arxiv__ord010__lsp1e-4/
      epoch005.pt                     # every 5 epochs
      epoch010.pt
      ...
      epoch030.pt                     # ← use this for interpretation
      windows0005M.pt                 # every 5M training windows consumed
      train_log.jsonl                 # loss per epoch {indep, sp, total, ETA}
    pythia1b4__arxiv__ord010__lsp1e-5/
    ...
  timing.jsonl                        # wall-clock time per stage
```

---

## Checkpoint format

Each `.pt` file contains:

```python
ckpt = torch.load("epoch030.pt", map_location="cpu")

ckpt["model_state"]["g_raw"]       # [256, 256]  mixing matrix
ckpt["model_state"]["B_list.0"]    # [256, 256]  lag-1 transition matrix
ckpt["model_state"]["B_list.1"]    # [256, 256]  lag-2 transition matrix
# ... up to B_list.{model_order-1}
ckpt["model_state"]["mask"]        # [256, 256]  fixed lower-block-triangular

ckpt["cfg"]   # {"model_order", "lambda_sp", "pca_components", "n_layers", "z_per_layer"}
ckpt["losses"]  # {"indep", "sp", "total"} at final epoch
```

Load for interpretation:
```python
import torch, numpy as np

ckpt = torch.load("epoch030.pt", map_location="cpu")
g    = ckpt["model_state"]["g_raw"]          # mixing matrix
f    = torch.linalg.inv(g)                   # demixing = encoder

acts_pca   = np.load("pythia1b4__arxiv.npy", mmap_mode="r")
boundaries = np.load("pythia1b4__arxiv.boundaries.npy")
categories = np.load("pythia1b4__arxiv.categories.npy", allow_pickle=True)

Z = torch.from_numpy(acts_pca[:]) @ f.T     # [N_tokens, 256] concept activations
```

---

## What to do next (interpretation)

Once training is done, the three evaluation tasks are:

**1. Category × layer heatmap** — for each passage compute mean activation per layer, group by label, plot `[categories × layers]`. Expected: Layer 0 high for math.LO, Layer 3 high for q-bio.

**2. Concentration (Gini coefficient)** — for each concept compute how narrowly it fires across categories. Expected: Layer 0 concepts have low Gini (fire everywhere), Layer 3 high Gini (fire in one domain).

**3. Cross-level concept links** — use the B matrices to find which Layer 0 concepts causally predict Layer 3 concepts. Then auto-label each concept using TF-IDF over its top activating passages, or via Claude API. Expected output:

```
L0_concept_7  "logical quantification"  [math.LO, math.CA]
      │  B-weight: 0.084
      ▼
L3_concept_12  "stochastic portfolio optimization"  [q-fin.MF, econ.GN]
```

Use the **final epoch checkpoint** (`epoch030.pt`) for all interpretation. Compare results across `model_order` values — larger order should produce cleaner hierarchy separation.