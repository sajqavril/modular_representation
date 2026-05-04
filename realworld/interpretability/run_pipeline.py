"""
Hierarchical SAE Pipeline
=========================
Stage 1 : Extract LLM activations (Pythia-1.4B @ layer 16, Llama-3-8B @ layer 20)
Stage 2 : PCA compression (fit on AMPS, apply to all datasets)
Stage 3 : Train linear hierarchical VAR model across model orders {1,2,3,5,10}

Model
─────
Input  : a window of (model_order + 1) consecutive token activations
         X = [x_t, x_{t-1}, ..., x_{t-p}]  where p = model_order
Latents: Z_t = X_t @ f.T   (f = inv(g), g is the mixing matrix)
VAR    : E_t = Z_t - Σ_{k=1}^{p} Z_{t-k} @ Bk.T
         Each Bk is an independent [z_dim × z_dim] matrix sharing the same
         lower-block-triangular hierarchy mask.
Loss   : reconstruction (skipped since x_dim=z_dim) + L1 independence on E_t + logdet(g) + sparsity on
         inter-layer blocks of every Bk.

SLURM   : designed for --array=0-19, each task_id maps to one
           (model, dataset, model_order) combination (see mapping table below).
Hardware: single A6000 / L40S (48 GB) per job
Cache   : respects HF_HOME / HF_DATASETS_CACHE env vars

Checkpoint naming
─────────────────
Epoch-based  : epoch005.pt, epoch010.pt, …, epoch030.pt   (every 5 epochs)
Window-based : windows0005M.pt, windows0010M.pt, …         (every 5M windows)
Both schemes run simultaneously and are independent.

Total checkpoint count (worst case)
────────────────────────────────────
  2 models × 2 datasets × 5 model_orders = 20 runs
  Per run: 6 epoch checkpoints + N window checkpoints
  Storage: each checkpoint ~few MB → well under 10 GB total

Timing log
──────────
  output_dir/timing.jsonl — one JSON line per timed event:
  {event, model, dataset, model_order, elapsed_sec, elapsed_hms, wall_time, ...extras}
"""

import os
import gc
import json
import math
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import IncrementalPCA
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset


# ════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    # ── paths ───────────────────────────────────────────────────────────────
    output_dir: str = "./hierarchical_sae_outputs"

    # ── models ──────────────────────────────────────────────────────────────
    models: List[dict] = field(default_factory=lambda: [
        {"name": "EleutherAI/pythia-1.4b",
         "layer": 16, "d_model": 2048, "tag": "pythia1b4"},
        {"name": "meta-llama/Meta-Llama-3-8B",
         "layer": 20, "d_model": 4096, "tag": "llama3_8b"},
    ])

    # ── datasets ────────────────────────────────────────────────────────────
    datasets: List[dict] = field(default_factory=lambda: [
    {
        # Dataset 1: arXiv — category labels define hierarchy
        # math.LO=L0, math.CA=L1, cs.LG=L2, q-bio=L3
        "tag": "arxiv",
        "local_path": "/data/user_data/jiaqisun/arxiv_metadata.json",
        "text_field": ["title", "abstract"],
        "min_chars": 100,
        "n_passages": 80_000,
    },
    {
        # Dataset 2: AMPS — difficulty/module labels define hierarchy
        # arithmetic=L0, calculus=L2, kl_divergence=L3
        "tag": "amps",
        "hf_path": "sarahpann/AMPS",
        "hf_split": "train",
        "text_field": ["problem", "step_by_step"],
        "min_chars": 100,
        "n_passages": 80_000,
    },
    {
        # Dataset 3: Wikipedia — category depth defines hierarchy
        # Mathematics=L0, Linear algebra=L1, Matrix decomposition=L3
        "tag": "wikipedia",
        "hf_path": "wikimedia/wikipedia",
        "hf_name": "20231101.en",
        "hf_split": "train",
        "text_field": ["title", "text"],
        "min_chars": 200,
        "n_passages": 80_000,
    },
])

    # ── extraction ──────────────────────────────────────────────────────────
    passage_max_tokens: int = 512
    extraction_batch_size: int = 8
    dtype: str = "bfloat16"

    # ── PCA ─────────────────────────────────────────────────────────────────
    pca_components: int = 256           # fit on AMPS, reuse for arXiv
    pca_batch_size: int = 4096

    # ── hierarchical model ──────────────────────────────────────────────────
    n_layers: int = 4
    z_per_layer: int = 64              # z_dim = 256
    # model_orders: VAR order p — number of lag matrices B1..Bp.
    # Input window = model_order + 1 consecutive adjacent tokens.
    model_orders: List[int] = field(default_factory=lambda: [10, 50, 100])

    # ── training ────────────────────────────────────────────────────────────
    train_batch_size: int = 512
    lr: float = 1e-3
    n_epochs: int = 30
    # lambda_indep is fixed at 1.0 — no recon loss since d_pca == z_dim
    # means g and f are exact inverses, reconstruction is identity.
    lambda_indep: float = 1.0
    # lambda_sp is swept; lambda_sp_values drives the SLURM grid
    lambda_sp_values: List[float] = field(
        default_factory=lambda: [1e-4, 1e-5])

    # checkpointing
    save_every_epochs: int = 5         # save every N epochs
    save_every_windows: int = 5_000_000  # also save every M windows consumed

    @property
    def z_dim(self):
        return self.n_layers * self.z_per_layer


CFG = Config()

# ════════════════════════════════════════════════════════════════════════════
# SLURM TASK → (model, dataset, model_order, lambda_sp) MAPPING
# ════════════════════════════════════════════════════════════════════════════
#
# 2 models × 2 datasets × 3 model_orders × 2 lambda_sp = 24 combinations
# Use --array=0-23
#
# task_id  model        dataset              model_order  lambda_sp
# ───────  ───────────  ───────────────────  ───────────  ─────────
#   0      pythia1b4    amps_khan             10           1e-4
#   1      pythia1b4    amps_khan             10           1e-5
#   2      pythia1b4    amps_khan             50           1e-4
#   3      pythia1b4    amps_khan             50           1e-5
#   4      pythia1b4    amps_khan            100           1e-4
#   5      pythia1b4    amps_khan            100           1e-5
#   6      pythia1b4    redpajama_arxiv       10           1e-4
#   7      pythia1b4    redpajama_arxiv       10           1e-5
#   8      pythia1b4    redpajama_arxiv       50           1e-4
#   9      pythia1b4    redpajama_arxiv       50           1e-5
#  10      pythia1b4    redpajama_arxiv      100           1e-4
#  11      pythia1b4    redpajama_arxiv      100           1e-5
#  12      llama3_8b    amps_khan             10           1e-4
#  13      llama3_8b    amps_khan             10           1e-5
#  14      llama3_8b    amps_khan             50           1e-4
#  15      llama3_8b    amps_khan             50           1e-5
#  16      llama3_8b    amps_khan            100           1e-4
#  17      llama3_8b    amps_khan            100           1e-5
#  18      llama3_8b    redpajama_arxiv       10           1e-4
#  19      llama3_8b    redpajama_arxiv       10           1e-5
#  20      llama3_8b    redpajama_arxiv       50           1e-4
#  21      llama3_8b    redpajama_arxiv       50           1e-5
#  22      llama3_8b    redpajama_arxiv      100           1e-4
#  23      llama3_8b    redpajama_arxiv      100           1e-5

def task_id_to_config(task_id: int, cfg: Config):
    combos = [
        (m, d, order, lsp)
        for m in cfg.models
        for d in cfg.datasets
        for order in cfg.model_orders
        for lsp in cfg.lambda_sp_values
    ]
    if task_id >= len(combos):
        raise ValueError(
            f"task_id={task_id} out of range "
            f"(max={len(combos)-1} for {len(combos)} combinations)"
        )
    return combos[task_id]


# ════════════════════════════════════════════════════════════════════════════
# TIMING LOGGER
# ════════════════════════════════════════════════════════════════════════════

def _fmt_hms(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}h{m:02d}m{s:05.2f}s"


class Timer:
    """
    Append-only JSONL timing log.
    Each stop() writes one line:
      {event, model, dataset, lag, elapsed_sec, elapsed_hms, wall_time, **extra}
    """
    def __init__(self, log_path: Path,
                 model_tag: str = "",
                 ds_tag: str = "",
                 lag: Optional[int] = None):
        self.log_path  = Path(log_path)
        self.model_tag = model_tag
        self.ds_tag    = ds_tag
        self.lag       = lag
        self._t0: Optional[float] = None
        self._label: str = ""

    def start(self, label: str):
        self._label = label
        self._t0    = time.perf_counter()
        print(f"  [TIME ▶] {label}", flush=True)

    def stop(self, extra: Optional[dict] = None) -> float:
        elapsed = time.perf_counter() - self._t0
        entry = {
            "event":       self._label,
            "model":       self.model_tag,
            "dataset":     self.ds_tag,
            "model_order": self.lag,   # stores model_order
            "elapsed_sec": round(elapsed, 3),
            "elapsed_hms": _fmt_hms(elapsed),
            "wall_time":   time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        if extra:
            entry.update(extra)
        with open(self.log_path, "a") as fh:
            fh.write(json.dumps(entry) + "\n")
        print(f"  [TIME ■] {self._label} → {_fmt_hms(elapsed)}", flush=True)
        return elapsed


# ════════════════════════════════════════════════════════════════════════════
# PATH HELPERS
# ════════════════════════════════════════════════════════════════════════════

def ensure_dir(p) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path

def act_path(out: Path, mtag: str, dtag: str) -> Path:
    return out / "activations" / f"{mtag}__{dtag}.npy"

def bnd_path(out: Path, mtag: str, dtag: str) -> Path:
    return out / "activations" / f"{mtag}__{dtag}.boundaries.npy"

def pca_save_path(out: Path, mtag: str) -> Path:
    return out / "pca" / f"{mtag}__pca.npz"

def ckpt_dir(out: Path, mtag: str, dtag: str,
             model_order: int, lambda_sp: float) -> Path:
    lsp_str = f"{lambda_sp:.0e}".replace("e-0", "e-")
    return out / "checkpoints" / f"{mtag}__{dtag}__ord{model_order:03d}__lsp{lsp_str}"

def timing_log_path(out: Path) -> Path:
    return out / "timing.jsonl"


# ════════════════════════════════════════════════════════════════════════════
# STAGE 1 — ACTIVATION EXTRACTION
# ════════════════════════════════════════════════════════════════════════════

def load_texts(ds_cfg: dict):
    # ── local JSONL fast path — bypasses all HuggingFace rate limits ─────────
    if "local_path" in ds_cfg:
        import json
        local_path = ds_cfg["local_path"]
        print(f"  Loading from local file: {local_path} ...", flush=True)
        text_field = ds_cfg["text_field"]
        min_chars  = ds_cfg.get("min_chars", 50)
        texts, categories = [], []
        with open(local_path, "r") as f:
            for line in f:
                row = json.loads(line)
                if isinstance(text_field, list):
                    text = " ".join(str(row.get(k, "")) for k in text_field).strip()
                else:
                    text = str(row.get(text_field, "")).strip()
                if len(text) < min_chars:
                    continue
                cat = row.get("categories", "")
                if isinstance(cat, list):
                    cat = " ".join(cat)
                texts.append(text)
                categories.append(str(cat).strip())
                if len(texts) >= ds_cfg["n_passages"]:
                    break
        n_with_cat = sum(1 for c in categories if c)
        print(f"  Collected {len(texts):,} passages  "
              f"({n_with_cat:,} with category labels).", flush=True)
        return texts, categories

    # ── HuggingFace streaming path (existing code below) ─────────────────────
    print(f"  Streaming: {ds_cfg['hf_path']} ...", flush=True)

    ds = load_dataset(
        ds_cfg["hf_path"],
        ds_cfg.get("hf_name", None),
        split=ds_cfg["hf_split"],
        streaming=True,
        trust_remote_code=True,
    )

    min_chars  = ds_cfg.get("min_chars", 50)
    text_field = ds_cfg["text_field"]
    texts      = []
    categories = []

    for row in ds:
        # optional source filter (e.g. old RedPajama meta.source)
        if "filter_key" in ds_cfg:
            meta   = row.get(ds_cfg["filter_key"], {})
            source = meta.get("source", "") if isinstance(meta, dict) else str(meta)
            if ds_cfg["filter_val"] not in source:
                continue

        # build text: single field or concatenation of multiple fields
        if isinstance(text_field, list):
            text = " ".join(str(row.get(f, "")) for f in text_field).strip()
        else:
            text = str(row.get(text_field, "")).strip()

        if len(text) < min_chars:
            continue

        # category metadata (arXiv only, empty string otherwise)
        cat = row.get("categories", "")
        if isinstance(cat, list):
            cat = " ".join(cat)
        categories.append(str(cat).strip())
        texts.append(text)

        if len(texts) >= ds_cfg["n_passages"]:
            break

    n_with_cat = sum(1 for c in categories if c)
    print(f"  Collected {len(texts):,} passages  "
          f"({n_with_cat:,} with category labels).", flush=True)
    return texts, categories

def get_transformer_blocks(model):
    if hasattr(model, "gpt_neox"):
        return model.gpt_neox.layers      # Pythia / GPT-NeoX
    if hasattr(model, "layers"):
        return model.layers               # Llama base model
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers         # LlamaForCausalLM-style
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h        # GPT-2-style
    raise ValueError(f"Could not find transformer blocks for {type(model)}")

@torch.no_grad()
def extract_activations_to_disk(texts, model_cfg, cfg, device, timer, ap: Path, bp: Path):
    print(f"  extract_activations received {len(texts)} texts", flush=True)
    if len(texts) == 0:
        raise ValueError("texts list is empty — check load_texts output")

    dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float32

    timer.start("load_model")
    tok = AutoTokenizer.from_pretrained(model_cfg["name"])
    tok.pad_token = tok.eos_token

    lm = AutoModel.from_pretrained(
        model_cfg["name"],
        torch_dtype=dtype,
        device_map="auto",
    )
    lm.eval()
    timer.stop({"model": model_cfg["name"]})

    d_model = model_cfg["d_model"]
    max_tokens = len(texts) * cfg.passage_max_tokens

    mm = np.lib.format.open_memmap(
        ap,
        mode="w+",
        dtype=np.float16,
        shape=(max_tokens, d_model),
    )

    boundaries = [0]
    total_tok = 0
    t_start = time.perf_counter()

    acts_cache = {}
    blocks = get_transformer_blocks(lm)
    hook_layer = model_cfg["layer"] - 1

    def save_activation_hook(module, inputs, output):
        acts_cache["h"] = output[0] if isinstance(output, tuple) else output

    handle = blocks[hook_layer].register_forward_hook(save_activation_hook)

    timer.start("forward_passes")

    try:
        for i in range(0, len(texts), cfg.extraction_batch_size):
            batch = texts[i: i + cfg.extraction_batch_size]

            enc = tok(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=cfg.passage_max_tokens,
            ).to(device)

            acts_cache.clear()
            _ = lm(**enc)

            if "h" not in acts_cache:
                raise RuntimeError(
                    f"Hook did not capture activations. "
                    f"hook_layer={hook_layer}, num_blocks={len(blocks)}"
                )

            h = acts_cache.pop("h")
            mask = enc["attention_mask"]

            for b in range(h.shape[0]):
                vlen = int(mask[b].sum().item())
                arr = h[b, :vlen, :].detach().to(torch.float16).cpu().numpy()

                mm[total_tok: total_tok + vlen] = arr
                total_tok += vlen
                boundaries.append(total_tok)

            del enc, h, mask
            torch.cuda.empty_cache()

            done = min(i + cfg.extraction_batch_size, len(texts))
            if (i // cfg.extraction_batch_size) % 100 == 0:
                el = time.perf_counter() - t_start
                rate = done / max(el, 1e-6)
                print(
                    f"    {done:>6,}/{len(texts):,} passages  "
                    f"{rate:.1f} pass/s  "
                    f"{total_tok/1e6:.2f}M tokens",
                    flush=True,
                )

    finally:
        handle.remove()

    timer.stop({"total_passages": len(texts), "total_tokens": total_tok})

    mm.flush()
    trimmed = np.asarray(mm[:total_tok])
    np.save(ap, trimmed)
    np.save(bp, np.array(boundaries, dtype=np.int64))

    del mm, trimmed, lm
    gc.collect()
    torch.cuda.empty_cache()

    print(f"  Saved activations shape: ({total_tok}, {d_model})", flush=True)
    print(f"  Size: {total_tok * d_model * 2 / 1e9:.2f} GB float16", flush=True)

    return total_tok


def run_extraction(cfg: Config, device: torch.device):
    out = ensure_dir(cfg.output_dir)
    ensure_dir(out / "activations")

    for mc in cfg.models:
        for dc in cfg.datasets:
            ap = act_path(out, mc["tag"], dc["tag"])
            bp = bnd_path(out, mc["tag"], dc["tag"])
            if ap.exists() and bp.exists():
                print(f"[SKIP] {ap.name} already exists.")
                continue

            print(f"\n── Extract: {mc['tag']} × {dc['tag']} ──")
            timer = Timer(timing_log_path(out), mc["tag"], dc["tag"])

            timer.start("load_texts")
            texts, categories = load_texts(dc)
            timer.stop({"n_passages": len(texts)})

            cp = ap.with_suffix(".categories.npy")

            timer.start("extract_and_save_to_disk")
            total_tok = extract_activations_to_disk(texts, mc, cfg, device, timer, ap, bp)
            np.save(cp, np.array(categories, dtype=object))
            timer.stop({"path": str(ap), "total_tokens": int(total_tok)})

            print(f"  Saved → {ap}")
            print(f"  Saved → {bp}")
            print(f"  Saved → {cp}")



# ════════════════════════════════════════════════════════════════════════════
# STAGE 2 — PCA  (fit on AMPS only, then reuse)
# ════════════════════════════════════════════════════════════════════════════

def fit_or_load_pca(cfg: Config, model_tag: str) -> IncrementalPCA:
    out = Path(cfg.output_dir)
    p   = pca_save_path(out, model_tag)
    ensure_dir(p.parent)

    if p.exists():
        print(f"[SKIP] PCA exists: {p.name}")
        data = np.load(p)
        pca  = IncrementalPCA(n_components=cfg.pca_components)
        pca.components_             = data["components"]
        pca.mean_                   = data["mean"]
        pca.explained_variance_     = data["explained_variance"]
        pca.explained_variance_ratio_ = data["explained_variance_ratio"]
        return pca

    ap = act_path(out, model_tag, "amps")
    if not ap.exists():
        raise FileNotFoundError(f"AMPS activations missing: {ap}")

    print(f"\n── PCA fit: {model_tag} on AMPS ──")
    timer = Timer(timing_log_path(out), model_tag, "amps")
    timer.start("pca_fit")

    acts = np.load(ap, mmap_mode="r")
    pca  = IncrementalPCA(n_components=cfg.pca_components)
    n    = acts.shape[0]
    for s in range(0, n, cfg.pca_batch_size):
        pca.partial_fit(acts[s: s + cfg.pca_batch_size])
        if s % (cfg.pca_batch_size * 20) == 0:
            print(f"  {min(s+cfg.pca_batch_size,n):,}/{n:,}", flush=True)

    var = float(pca.explained_variance_ratio_.sum())
    timer.stop({"n_components": cfg.pca_components,
                "variance_explained": round(var, 4)})
    print(f"  Variance explained: {var:.4f}")

    np.savez(p,
             components=pca.components_,
             mean=pca.mean_,
             explained_variance=pca.explained_variance_,
             explained_variance_ratio=pca.explained_variance_ratio_)
    return pca


def apply_pca_batched(acts: np.ndarray, pca: IncrementalPCA) -> np.ndarray:
    k   = pca.components_.shape[0]
    out = np.empty((acts.shape[0], k), dtype=np.float32)
    bs  = 8192
    for i in range(0, acts.shape[0], bs):
        out[i: i+bs] = pca.transform(acts[i: i+bs]).astype(np.float32)
    return out


# ════════════════════════════════════════════════════════════════════════════
# STAGE 3A — DATASET
# ════════════════════════════════════════════════════════════════════════════

class VARWindowDataset(Dataset):
    """
    Yields windows of (model_order + 1) consecutive token activations
    from a single passage, never crossing boundaries.

    Input tokens are adjacent (lag=1 at the data level).
    The model_order p controls how many past latents are used in the VAR:
      window = [x_t, x_{t-1}, ..., x_{t-p}]   shape: [p+1, d_pca]
    where x_t is the current token and x_{t-k} are k steps back.

    __getitem__ returns:
      X_now  : [d_pca]        current token x_t
      X_past : [p, d_pca]     past tokens [x_{t-1}, ..., x_{t-p}] in order
    """
    def __init__(self, acts_pca: np.ndarray,
                 boundaries: np.ndarray, model_order: int):
        self.acts        = acts_pca
        self.model_order = model_order
        p                = model_order

        # anchor indices: t must satisfy t-p >= passage_start
        anchors = []
        for i in range(len(boundaries) - 1):
            s, e = int(boundaries[i]), int(boundaries[i+1])
            if e - s <= p:          # passage too short for this order
                continue
            # valid anchor range: t in [s+p, e)
            anchors.append(np.arange(s + p, e, dtype=np.int64))

        self.anchors = np.concatenate(anchors, axis=0)
        print(f"  VARWindowDataset order={model_order:>2d}: "
              f"{len(self.anchors):>12,} windows  "
              f"({len(boundaries)-1:,} passages)", flush=True)

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, idx):
        t   = int(self.anchors[idx])
        p   = self.model_order
        # current token
        X_now  = torch.from_numpy(self.acts[t].copy())           # [d_pca]
        # past tokens: [x_{t-1}, x_{t-2}, ..., x_{t-p}]
        past   = np.stack([self.acts[t - k] for k in range(1, p + 1)])
        X_past = torch.from_numpy(past.copy())                    # [p, d_pca]
        return X_now, X_past


# ════════════════════════════════════════════════════════════════════════════
# STAGE 3B — MODEL
# ════════════════════════════════════════════════════════════════════════════

class HierarchicalSAE(nn.Module):
    """
    Linear VAR generative model with hierarchy mask on every lag matrix.

    Parameters
    ──────────
    d_pca        : PCA dimension (input/output)
    n_layers     : number of hierarchy layers
    z_per_layer  : concepts per layer  →  z_dim = n_layers * z_per_layer
    model_order  : VAR order p — number of lag matrices B1..Bp

    Architecture
    ────────────
    g            : [z_dim, d_pca]   mixing matrix  (learned)
    B_list       : p × [z_dim, z_dim]  lag matrices (learned, one per lag k)
    mask         : [z_dim, z_dim]   lower-block-triangular, fixed, shared by all Bk

    Forward
    ───────
    Given X_now [batch, d_pca] and X_past [batch, p, d_pca]:
      Z_t   = X_now  @ f.T                      encode current token
      Z_{tk} = X_past[:,k,:] @ f.T              encode each past token
      E_t   = Z_t - Σ_{k=1}^{p} Z_{tk} @ Bk.T  VAR residual
    Losses: reconstruction + L1(E_t) + logdet(g) + inter-layer sparsity on every Bk
    """
    def __init__(self, d_pca: int, n_layers: int,
                 z_per_layer: int, model_order: int):
        super().__init__()
        self.d_pca       = d_pca
        self.layer       = n_layers
        self.z_per_layer = z_per_layer
        self.z_dim       = n_layers * z_per_layer
        self.model_order = model_order

        self.g_raw = nn.Parameter(
            torch.randn(self.z_dim, d_pca) / math.sqrt(d_pca)
        )

        # One independent B matrix per lag order k=1..p
        # All share the same hierarchy mask (registered as a buffer)
        self.B_list = nn.ParameterList([
            nn.Parameter(torch.zeros(self.z_dim, self.z_dim))
            for _ in range(model_order)
        ])

        # Lower-block-triangular mask:
        # row block l can only depend on column blocks 0..l (not l+1..)
        mask = torch.ones(self.z_dim, self.z_dim)
        for _l in range(n_layers):
            mask[_l * z_per_layer: (_l + 1) * z_per_layer,
                 (_l + 1) * z_per_layer:] = 0.
        self.register_buffer("mask", mask)

    def _inv_g(self, g):
        return (torch.linalg.inv(g) if g.shape[0] == g.shape[1]
                else torch.linalg.pinv(g))

    def forward(self, X_now, X_past):
        """
        X_now  : [batch, d_pca]
        X_past : [batch, p, d_pca]   p = model_order
        """
        g = self.g_raw
        f = self._inv_g(g)

        # Encode current token
        Z_t = X_now @ f.T                              # [batch, z_dim]

        # Encode all past tokens
        # X_past[:,k,:] is x_{t-(k+1)}  (k=0 → lag 1, k=p-1 → lag p)
        Z_past = X_past @ f.T                          # [batch, p, z_dim]

        # No reconstruction loss: d_pca == z_dim so g is square and
        # Z @ g = X exactly by construction (f = inv(g)).
        # We return 0. as a placeholder to keep the return signature stable.
        loss_recon = torch.tensor(0., device=X_now.device)

        # VAR residual: E_t = Z_t - Σ_k Z_{t-k} @ Bk.T
        E_t = Z_t.clone()
        for k, B_raw in enumerate(self.B_list):
            Bk   = B_raw * self.mask               # apply hierarchy mask
            E_t  = E_t - Z_past[:, k, :] @ Bk.T

        # Independence loss on residual
        logdet = (torch.slogdet(g)[1] if g.shape[0] == g.shape[1]
                  else 0.5 * torch.slogdet(g @ g.T)[1])
        loss_indep = E_t.abs().mean() + logdet

        # Sparsity: penalise inter-layer blocks of every Bk
        loss_sp = torch.tensor(0., device=X_now.device)
        for B_raw in self.B_list:
            Bk = B_raw * self.mask
            for _l in range(1, self.layer):
                block = Bk[_l * self.z_per_layer: (_l + 1) * self.z_per_layer,
                           (_l - 1) * self.z_per_layer: _l * self.z_per_layer]
                loss_sp += block.abs().mean()
        # normalise by (n_inter_layer_pairs × model_order)
        loss_sp /= max((self.layer - 1) * self.model_order, 1)

        # Return masked B matrices for inspection
        Bs = [B_raw * self.mask for B_raw in self.B_list]
        # loss_recon=0. always (exact inverse), kept in tuple for API compat
        return torch.tensor(0., device=X_now.device), loss_indep, loss_sp, Bs, g


# ════════════════════════════════════════════════════════════════════════════
# STAGE 3C — TRAINING
# ════════════════════════════════════════════════════════════════════════════

def _save_ckpt(cd, name, model, optimizer,
               epoch, windows_consumed, losses, cfg, model_order, lambda_sp=None):
    path = cd / name
    torch.save({
        "epoch":             epoch,
        "windows_consumed":  windows_consumed,
        "model_state":       model.state_dict(),
        "optimizer_state":   optimizer.state_dict(),
        "losses":            losses,
        "cfg": {
            "model_order":    model_order,
            "lambda_sp":      lambda_sp,
            "pca_components": cfg.pca_components,
            "n_layers":       cfg.n_layers,
            "z_per_layer":    cfg.z_per_layer,
        },
    }, path)
    print(f"    ✓ ckpt → {name}", flush=True)


def train_one_config(acts_pca, boundaries, model_order, lambda_sp,
                     cfg, cd, device, model_tag, ds_tag):
    ensure_dir(cd)
    train_log  = cd / "train_log.jsonl"
    timer      = Timer(timing_log_path(Path(cfg.output_dir)),
                       model_tag, ds_tag, model_order)

    timer.start("build_dataset")
    dataset   = VARWindowDataset(acts_pca, boundaries, model_order)
    n_windows = len(dataset)
    timer.stop({"n_windows": n_windows})

    loader = DataLoader(
        dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    windows_per_epoch = (n_windows // cfg.train_batch_size) * cfg.train_batch_size

    model = HierarchicalSAE(
        d_pca=cfg.pca_components,
        n_layers=cfg.n_layers,
        z_per_layer=cfg.z_per_layer,
        model_order=model_order,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    opt   = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.n_epochs)

    print(f"\n  Training: {model_tag} × {ds_tag} × order={model_order} × lsp={lambda_sp:.0e}\n"
          f"  params={n_params:,}  "
          f"windows/epoch={windows_per_epoch:,}  "
          f"epochs={cfg.n_epochs}", flush=True)

    n_win_ckpts = (windows_per_epoch * cfg.n_epochs) // cfg.save_every_windows
    print(f"  Expected checkpoints: "
          f"{cfg.n_epochs // cfg.save_every_epochs} epoch-based  +  "
          f"~{n_win_ckpts} window-based", flush=True)

    wins_since_save  = 0
    windows_consumed = 0

    timer.start("training")
    t_train = time.perf_counter()

    for epoch in range(1, cfg.n_epochs + 1):
        t_ep = time.perf_counter()
        model.train()
        L = {"indep": 0., "sp": 0., "total": 0.}
        nb = 0

        for X_now, X_past in loader:
            X_now  = X_now.to(device)                 # [batch, d_pca]
            X_past = X_past.to(device)                 # [batch, p, d_pca]

            # no recon loss: d_pca==z_dim so g is square and Z@g = X exactly
            _, li, ls, _, _ = model(X_now, X_past)
            loss = li + lambda_sp * ls   # lambda_indep=1.0 fixed, recon=0

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            L["indep"]  += li.item()
            L["sp"]     += ls.item()
            L["total"]  += loss.item()
            nb += 1

            windows_consumed += cfg.train_batch_size
            wins_since_save  += cfg.train_batch_size

            # window-based checkpoint
            if wins_since_save >= cfg.save_every_windows:
                avg = {k: v/nb for k, v in L.items()}
                _save_ckpt(cd,
                           f"windows{windows_consumed//1_000_000:04d}M.pt",
                           model, opt, epoch, windows_consumed,
                           avg, cfg, model_order, lambda_sp)
                wins_since_save = 0

        sched.step()
        for k in L:
            L[k] /= nb

        ep_t  = time.perf_counter() - t_ep
        tot_t = time.perf_counter() - t_train
        eta   = (cfg.n_epochs - epoch) * ep_t

        log = {
            "epoch":              epoch,
            "model_order":        model_order,
            "lambda_sp":          lambda_sp,
            "windows_consumed":   windows_consumed,
            "elapsed_epoch_sec":  round(ep_t, 3),
            "elapsed_total_sec":  round(tot_t, 3),
            "eta_sec":            round(eta, 1),
            "eta_hms":            _fmt_hms(eta),
            **L,
        }
        with open(train_log, "a") as fh:
            fh.write(json.dumps(log) + "\n")

        print(f"  epoch {epoch:3d}/{cfg.n_epochs}  "
              f"indep={L['indep']:.4f}  "
              f"sp={L['sp']:.4f}  "
              f"wins={windows_consumed/1e6:.1f}M  "
              f"ep={_fmt_hms(ep_t)}  "
              f"ETA={_fmt_hms(eta)}", flush=True)

        # epoch-based checkpoint
        if epoch % cfg.save_every_epochs == 0 or epoch == cfg.n_epochs:
            _save_ckpt(cd, f"epoch{epoch:03d}.pt",
                       model, opt, epoch, windows_consumed,
                       L, cfg, model_order, lambda_sp)

    timer.stop({
        "total_windows":   windows_consumed,
        "total_windows_M": round(windows_consumed / 1e6, 2),
    })


# ════════════════════════════════════════════════════════════════════════════
# TOP-LEVEL ENTRY POINT PER TASK
# ════════════════════════════════════════════════════════════════════════════

def run_task(model_cfg, ds_cfg, model_order, lambda_sp, cfg, device):
    out      = Path(cfg.output_dir)
    mtag, dtag = model_cfg["tag"], ds_cfg["tag"]
    timer    = Timer(timing_log_path(out), mtag, dtag, model_order)
    cd       = ckpt_dir(out, mtag, dtag, model_order, lambda_sp)

    if (cd / f"epoch{cfg.n_epochs:03d}.pt").exists():
        print(f"[SKIP] Complete: {cd.name}")
        return

    # PCA (shared across tasks for same model; safe to call in parallel
    # because it early-returns if the .npz already exists)
    timer.start("pca_load_or_fit")
    pca = fit_or_load_pca(cfg, mtag)
    timer.stop()

    # load & transform activations
    ap = act_path(out, mtag, dtag)
    bp = bnd_path(out, mtag, dtag)
    if not ap.exists():
        raise FileNotFoundError(
            f"Activations missing: {ap}\nRun --stage extract first."
        )

    timer.start("pca_transform")
    acts_raw   = np.load(ap, mmap_mode="r")
    acts_pca   = apply_pca_batched(acts_raw, pca)
    boundaries = np.load(bp)
    timer.stop({"shape": list(acts_pca.shape)})

    train_one_config(acts_pca, boundaries, model_order, lambda_sp,
                     cfg, cd, device, mtag, dtag)


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical SAE — SLURM array pipeline"
    )
    parser.add_argument(
        "--stage",
        choices=["extract", "train", "all"],
        default="train",
        help=(
            "extract : activation extraction only (run once before array)\n"
            "train   : training for this task_id\n"
            "all     : extract then train"
        ),
    )
    parser.add_argument(
        "--task_id", type=int, default=None,
        help=(
            "SLURM array task id (0-19).  "
            "Falls back to $SLURM_ARRAY_TASK_ID.  "
            "If neither set, all 20 combos run serially."
        ),
    )
    parser.add_argument("--output_dir", default=CFG.output_dir)
    args = parser.parse_args()

    CFG.output_dir = args.output_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device : {device}", flush=True)
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"GPU    : {p.name}  ({p.total_memory/1e9:.1f} GB)", flush=True)

    # resolve task_id
    task_id = args.task_id
    if task_id is None:
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", -1))

    if args.stage in ("extract", "all"):
        print("\n" + "="*60 + "\nSTAGE 1: EXTRACTION\n" + "="*60)
        run_extraction(CFG, device)

    if args.stage in ("train", "all"):
        print("\n" + "="*60 + "\nSTAGE 3: TRAINING\n" + "="*60)
        if task_id >= 0:
            mc, dc, order, lsp = task_id_to_config(task_id, CFG)
            print(f"SLURM task_id={task_id}  →  "
                  f"{mc['tag']} × {dc['tag']} × order={order} × lsp={lsp:.0e}")
            run_task(mc, dc, order, lsp, CFG, device)
        else:
            n = (len(CFG.models) * len(CFG.datasets)
                 * len(CFG.model_orders) * len(CFG.lambda_sp_values))
            print(f"No task_id — running all {n} combos serially.")
            for tid in range(n):
                mc, dc, order, lsp = task_id_to_config(tid, CFG)
                run_task(mc, dc, order, lsp, CFG, device)

    print("\n✓ Done.", flush=True)


if __name__ == "__main__":
    main()