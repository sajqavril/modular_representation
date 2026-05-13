#!/usr/bin/env python3
"""
rw_rq2_probe.py — RW-RQ2: Label-efficient downstream prediction.

Labels use EXACTLY the same sources as the interpretation script:
  arxiv     : categories field from local JSON (arxiv_metadata.json)
  amps      : re-streamed from HuggingFace sarahpann/AMPS, labeled by text keywords
  wikipedia : re-streamed from HuggingFace wikimedia/wikipedia, labeled by title+length
  (No .categories.npy dependency — those files may not exist)

Reps : Raw PCA | MoRe L0 | MoRe L1 | MoRe L2 | MoRe L3 | MoRe all
Task : Linear probe to predict tier label from frozen representation
"""

from __future__ import annotations
import argparse, csv, json, math, time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.decomposition import IncrementalPCA


# =============================================================================
# Paths
# =============================================================================

def act_path(out, mtag, dtag):
    return Path(out) / "activations" / f"{mtag}__{dtag}.npy"

def bnd_path(out, mtag, dtag):
    return Path(out) / "activations" / f"{mtag}__{dtag}.boundaries.npy"

def cat_path(out, mtag, dtag):   # may not exist — kept for compat
    return Path(out) / "activations" / f"{mtag}__{dtag}.categories.npy"

def pca_path(out, mtag):
    return Path(out) / "pca" / f"{mtag}__pca.npz"

def ckpt_dir(out, mtag, dtag, order, lsp):
    lsp_str = f"{lsp:.0e}".replace("e-0", "e-")
    return Path(out) / "checkpoints" / f"{mtag}__{dtag}__ord{order:03d}__lsp{lsp_str}"


# =============================================================================
# Label generation — ported exactly from interpretation script
# =============================================================================

# arXiv category → abstraction tier
ARXIV_TIER = {
    "math.LO": 0, "math.CT": 0, "cs.LO": 0,
    "math.CA": 1, "math.GR": 1, "math.AG": 1, "math.NT": 1,
    "cs.CC": 1, "math.FA": 1,
    "cs.LG": 2, "cs.IT": 2, "stat.TH": 2, "cs.AI": 2,
    "stat.ML": 2, "cs.CV": 2, "cs.NE": 2,
    "q-bio": 3, "econ": 3, "cond-mat": 3, "astro-ph": 3,
    "q-fin": 3, "physics": 3,
}

AMPS_TIER_KEYWORDS = [
    # most specific first — same order as interpretation script
    (3, ["kl divergence", "kullback", "stochastic", "fourier", "laplace",
         "manifold", "topology", "eigenvalue", "eigenvector", "determinant",
         "markov", "bayesian", "entropy"]),
    (2, ["derivative", "integral", "calculus", "differential equation",
         "matrix", "vector space", "linear transform", "gradient",
         "taylor", "convergence", "limit"]),
    (1, ["algebra", "equation", "polynomial", "factor", "quadratic",
         "trigonometry", "logarithm", "exponential", "inequality",
         "function", "graph"]),
    (0, ["add", "subtract", "multiply", "divide", "fraction", "decimal",
         "percent", "ratio", "arithmetic", "simplify", "evaluate"]),
]

def pca_path(out, mtag, dtag="amps"):
    specific = Path(out) / "pca" / f"{mtag}__{dtag}__pca.npz"
    if specific.exists():
        return specific
    fallback = Path(out) / "pca" / f"{mtag}__pca.npz"
    if fallback.exists():
        return fallback
    return specific   # return specific path for saving


def load_pca(out: str, mtag: str,
             dtag: str = "amps") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load PCA for (model, dataset). Priority:
      1. {mtag}__{dtag}__pca.npz  — dataset-specific (preferred)
      2. {mtag}__pca.npz          — legacy AMPS-fitted fallback
      3. Compute from activations  — if neither exists
    """
    p = pca_path(out, mtag, dtag)

    if p.exists():
        print(f"    Loading PCA: {p.name}", flush=True)
        data = np.load(p)
        return (data["components"].astype(np.float32),
                data["mean"].astype(np.float32))

    # ── compute PCA from activations ──────────────────────────────────────
    ap = act_path(out, mtag, dtag)
    if not ap.exists():
        raise FileNotFoundError(
            f"Cannot compute PCA: activations missing at {ap}"
        )

    print(f"    PCA not found — computing from {ap.name} ...", flush=True)
    acts = np.load(ap, mmap_mode="r")
    n_components = 256
    pca = IncrementalPCA(n_components=n_components)
    batch_size = 4096
    n = acts.shape[0]
    for s in range(0, n, batch_size):
        pca.partial_fit(acts[s: s + batch_size].astype(np.float32))
        if s % (batch_size * 20) == 0:
            print(f"      PCA fit: {min(s+batch_size, n):,}/{n:,}", flush=True)

    var = float(pca.explained_variance_ratio_.sum())
    print(f"    Variance explained: {var:.4f}", flush=True)

    # save for future runs
    save_path = Path(out) / "pca" / f"{mtag}__{dtag}__pca.npz"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(save_path,
             components=pca.components_,
             mean=pca.mean_,
             explained_variance=pca.explained_variance_,
             explained_variance_ratio=pca.explained_variance_ratio_)
    print(f"    Saved PCA → {save_path.name}", flush=True)

    return (pca.components_.astype(np.float32),
            pca.mean_.astype(np.float32))

def amps_label(text: str) -> str:
    """Exact copy from interpretation script."""
    t = text.lower()
    for tier, keywords in AMPS_TIER_KEYWORDS:
        for kw in keywords:
            if kw in t:
                return f"tier_{tier}"
    return "tier_1"   # default: algebra-level


def wikipedia_label(text: str, title: str = "") -> str:
    """Exact copy from interpretation script, returning tier string."""
    n_words = len(text.split())
    title_l = title.lower()
    specific_kw = ["theorem", "model", "equation", "algorithm", "formula",
                   "hypothesis", "lemma", "conjecture", "proof"]
    general_kw  = ["mathematics", "science", "history", "philosophy",
                   "logic", "language", "art"]
    if any(kw in title_l for kw in specific_kw):
        return "tier_3"
    if any(kw in title_l for kw in general_kw):
        return "tier_0"
    if n_words < 500:  return "tier_0"
    if n_words < 2000: return "tier_1"
    if n_words < 5000: return "tier_2"
    return "tier_3"


def arxiv_label(cat_str: str) -> Optional[str]:
    """Map arXiv category string to tier_0/1/2/3 using ARXIV_TIER, else top-level."""
    if not cat_str or cat_str in ("nan", "", "None"):
        return None
    primary = cat_str.strip().split()[0]
    # Try exact match first.
    if primary in ARXIV_TIER:
        return f"tier_{ARXIV_TIER[primary]}"
    # Try prefix match.
    prefix = primary.split(".")[0]
    for k, v in ARXIV_TIER.items():
        if k.startswith(prefix + ".") or k == prefix:
            return f"tier_{v}"
    # Fall back to top-level domain as label (not tier).
    return prefix if prefix else None


# =============================================================================
# Load labels — same sources as interpretation script
# =============================================================================

def load_labels_for_dataset(
    dtag: str,
    n_passages: int,
    output_dir: str,
    arxiv_local_path: str,
    max_stream: int = 0,   # 0 = load all
) -> List[Optional[str]]:
    """
    Return list of tier labels (or None) of length n_passages.
    Matches interpretation script label sources exactly.
    """
    labels = [None] * n_passages

    if dtag == "arxiv":
        # Load from local JSON — same as pipeline's load_texts.
        arxiv_path = Path(arxiv_local_path)
        if not arxiv_path.exists():
            print(f"  [WARN] arXiv JSON not found: {arxiv_path}. "
                  f"Try --arxiv_local_path.")
            return labels
        print(f"  Loading arXiv labels from {arxiv_path} ...", flush=True)
        idx = 0
        with open(arxiv_path) as f:
            for line in f:
                if idx >= n_passages: break
                try:
                    row = json.loads(line)
                    cat = row.get("categories", "")
                    if isinstance(cat, list):
                        cat = " ".join(cat)
                    labels[idx] = arxiv_label(str(cat).strip())
                except Exception:
                    pass
                idx += 1
        n_labeled = sum(1 for l in labels if l is not None)
        print(f"  arXiv: {n_labeled}/{n_passages} passages labeled.", flush=True)

    elif dtag == "amps":
        # Re-stream from HuggingFace — same as interpretation script.
        print(f"  Streaming AMPS labels ({n_passages} passages)...", flush=True)
        try:
            from datasets import load_dataset as hf_load
            ds = hf_load("sarahpann/AMPS", split="train",
                         streaming=True, trust_remote_code=True)
            idx = 0
            for row in ds:
                if idx >= n_passages: break
                prob = str(row.get("problem", "")).strip()
                step = str(row.get("step_by_step", "")).strip()
                text = (prob + " " + step).strip()
                labels[idx] = amps_label(text)
                idx += 1
            print(f"  AMPS: labeled {idx} passages.", flush=True)
        except Exception as e:
            print(f"  [WARN] AMPS streaming failed: {e}")

    elif dtag == "wikipedia":
        # Re-stream from HuggingFace — same as interpretation script.
        print(f"  Streaming Wikipedia labels ({n_passages} passages)...", flush=True)
        try:
            from datasets import load_dataset as hf_load
            ds = hf_load("wikimedia/wikipedia", "20231101.en",
                         split="train", streaming=True, trust_remote_code=True)
            idx = 0
            for row in ds:
                if idx >= n_passages: break
                title = str(row.get("title", "")).strip()
                text  = str(row.get("text",  "")).strip()
                labels[idx] = wikipedia_label(text, title)
                idx += 1
            print(f"  Wikipedia: labeled {idx} passages.", flush=True)
        except Exception as e:
            print(f"  [WARN] Wikipedia streaming failed: {e}")

    return labels


# =============================================================================
# Load PCA and checkpoint
# =============================================================================

def load_pca(out: str, mtag: str) -> Tuple[np.ndarray, np.ndarray]:
    p = pca_path(out, mtag)
    data = np.load(p)
    return data["components"].astype(np.float32), data["mean"].astype(np.float32)


def load_model_f(out: str, mtag: str, dtag: str,
                 order: int = 10, lsp: float = 1e-4,
                 epoch: int = 30) -> Optional[np.ndarray]:
    cd = ckpt_dir(out, mtag, dtag, order, lsp)
    ckpt_file = cd / f"epoch{epoch:03d}.pt"
    if not ckpt_file.exists():
        available = sorted(cd.glob("epoch*.pt"))
        if not available:
            print(f"  [WARN] No checkpoint in {cd}")
            return None
        ckpt_file = available[-1]
        print(f"  [INFO] Using: {ckpt_file.name}", flush=True)
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    g_raw = ckpt["model_state"]["g_raw"].float().numpy()
    try:
        f = np.linalg.inv(g_raw)
    except np.linalg.LinAlgError:
        f = np.linalg.pinv(g_raw)
    return f.astype(np.float32)


# =============================================================================
# Build passage-level representations
# =============================================================================

def pca_transform_batched(acts: np.ndarray, pca_comp: np.ndarray,
                           pca_mean: np.ndarray, bs: int = 8192) -> np.ndarray:
    out = np.empty((acts.shape[0], pca_comp.shape[0]), dtype=np.float32)
    for i in range(0, acts.shape[0], bs):
        chunk = acts[i:i+bs].astype(np.float32) - pca_mean
        out[i:i+bs] = chunk @ pca_comp.T
    return out


def mean_pool_passages(token_reps: np.ndarray,
                       boundaries: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n_passages = len(boundaries) - 1
    dim = token_reps.shape[1]
    out = np.zeros((n_passages, dim), dtype=np.float32)
    valid = []
    for i in range(n_passages):
        s, e = int(boundaries[i]), int(boundaries[i+1])
        if e > s:
            out[i] = token_reps[s:e].mean(axis=0)
            valid.append(i)
    return out, np.array(valid)


def build_representations(
    out: str, mtag: str, dtag: str,
    order: int = 10, lsp: float = 1e-4, epoch: int = 30,
    n_layers: int = 4, z_per_layer: int = 64,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Returns reps dict, boundaries, valid_passage_indices."""
    print(f"\n  Building reps: {mtag} x {dtag}", flush=True)
    t0 = time.perf_counter()

    acts_raw = np.load(act_path(out, mtag, dtag), mmap_mode="r")
    bounds   = np.load(bnd_path(out, mtag, dtag))
    pca_comp, pca_mean = load_pca(out, mtag)

    print(f"    PCA transform {acts_raw.shape} ...", flush=True)
    acts_pca = pca_transform_batched(acts_raw, pca_comp, pca_mean)
    del acts_raw

    pca_pooled, valid_idx = mean_pool_passages(acts_pca, bounds)
    f = load_model_f(out, mtag, dtag, order, lsp, epoch)

    reps = {"PCA": pca_pooled}

    if f is not None:
        Z_all = acts_pca @ f   # (total_tokens, z_dim)
        Z_pooled_all, _ = mean_pool_passages(Z_all, bounds)
        reps["MoRe_all"] = Z_pooled_all
        for layer_idx in range(n_layers):
            s = layer_idx * z_per_layer
            e = (layer_idx + 1) * z_per_layer
            Z_layer = acts_pca @ f[:, s:e]
            Z_layer_pooled, _ = mean_pool_passages(Z_layer, bounds)
            reps[f"MoRe_L{layer_idx}"] = Z_layer_pooled
        del Z_all, Z_pooled_all

    del acts_pca
    print(f"    {len(reps)} reps built in {time.perf_counter()-t0:.1f}s", flush=True)
    return reps, bounds, valid_idx


# =============================================================================
# Linear probe
# =============================================================================

def probe_one(Z, labels, n_labeled_list, n_seeds=5):
    results = {}
    scaler = StandardScaler()
    Z_scaled = scaler.fit_transform(Z)
    n_classes = len(np.unique(labels))

    for n_labeled in n_labeled_list:
        if n_labeled < n_classes or n_labeled >= len(labels) * 0.8:
            continue
        accs = []
        try:
            sss = StratifiedShuffleSplit(
                n_splits=n_seeds, train_size=n_labeled,
                test_size=min(2000, len(labels) - n_labeled), random_state=42)
            for tr, te in sss.split(Z_scaled, labels):
                clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
                clf.fit(Z_scaled[tr], labels[tr])
                accs.append(accuracy_score(labels[te], clf.predict(Z_scaled[te])))
        except Exception as e:
            print(f"      [WARN] n={n_labeled}: {e}")
            continue
        results[n_labeled] = {"mean": float(np.mean(accs)),
                               "std": float(np.std(accs)), "n_seeds": len(accs)}
    return results


# =============================================================================
# Main
# =============================================================================

def run_rq2(args):
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_labeled_list = [10, 25, 50, 100, 250, 500, 1000]
    all_results, rows = {}, []

    for mtag in args.models:
        for dtag in args.datasets:
            key = f"{mtag}__{dtag}"
            print(f"\n{'='*60}\n  {key}\n{'='*60}", flush=True)

            if not act_path(args.output_dir, mtag, dtag).exists():
                print(f"  [SKIP] Activations not found."); continue

            try:
                reps, bounds, valid_idx = build_representations(
                    args.output_dir, mtag, dtag,
                    order=args.model_order, lsp=args.lambda_sp,
                    epoch=args.epoch, n_layers=args.n_layers,
                    z_per_layer=args.z_per_layer,
                )
            except Exception as e:
                print(f"  [ERROR] {e}"); continue

            n_passages = len(bounds) - 1

            # Load labels using same sources as interpretation script.
            all_labels = load_labels_for_dataset(
                dtag, n_passages, args.output_dir,
                arxiv_local_path=args.arxiv_local_path,
            )

            # Select labels for valid (non-empty) passages.
            labels_valid = [all_labels[i] for i in valid_idx]
            counts = Counter(l for l in labels_valid if l is not None)
            print(f"  Label distribution: {dict(counts.most_common(10))}", flush=True)

            valid_classes = {k for k, v in counts.items() if v >= args.min_freq}
            if len(valid_classes) < 2:
                print(f"  [SKIP] <2 classes with >= {args.min_freq} passages. "
                      f"All counts: {dict(counts.most_common())}"); continue

            label_names = sorted(valid_classes)
            label2int = {l: i for i, l in enumerate(label_names)}
            final_idx = [i for i, l in enumerate(labels_valid) if l in valid_classes]
            final_labels = np.array([label2int[labels_valid[i]] for i in final_idx])
            print(f"  {len(label_names)} classes, {len(final_labels)} passages: "
                  f"{label_names}", flush=True)

            all_results[key] = {"label_names": label_names, "reps": {}}

            for rep_name, Z_full in reps.items():
                Z = Z_full[final_idx]
                print(f"\n    Probing: {rep_name}  shape={Z.shape}", flush=True)
                t0 = time.perf_counter()
                probe_res = probe_one(Z, final_labels, n_labeled_list, args.n_seeds)
                print(f"    Done {time.perf_counter()-t0:.1f}s:", flush=True)
                for n, r in probe_res.items():
                    print(f"      n={n:5d}: {r['mean']:.3f} ± {r['std']:.3f}", flush=True)

                all_results[key]["reps"][rep_name] = probe_res
                for n_labeled, r in probe_res.items():
                    rows.append({
                        "model": mtag, "dataset": dtag,
                        "representation": rep_name,
                        "n_labeled": n_labeled,
                        "accuracy_mean": r["mean"], "accuracy_std": r["std"],
                        "n_seeds": r["n_seeds"], "n_classes": len(label_names),
                        "label_names": "|".join(label_names),
                    })

            with open(out_dir / "probe_results.json", "w") as f2:
                json.dump(all_results, f2, indent=2)
            _write_csv(rows, out_dir / "probe_results.csv")

    _make_figures(all_results, out_dir)
    print(f"\nDone. Results: {out_dir.resolve()}", flush=True)


def _write_csv(rows, path):
    if not rows: return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def _make_figures(results, out_dir):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig_dir = out_dir / "figures"; fig_dir.mkdir(exist_ok=True)
        colors = {"PCA": "gray", "MoRe_L0": "#1f77b4", "MoRe_L1": "#ff7f0e",
                  "MoRe_L2": "#2ca02c", "MoRe_L3": "#d62728", "MoRe_all": "black"}
        for key, data in results.items():
            fig, ax = plt.subplots(figsize=(7, 5))
            for rep_name, pr in data["reps"].items():
                if not pr: continue
                ns = sorted(pr.keys())
                means = [pr[n]["mean"] for n in ns]
                stds  = [pr[n]["std"]  for n in ns]
                c = colors.get(rep_name, "purple")
                ax.plot(ns, means, marker="o", label=rep_name, color=c, linewidth=2)
                ax.fill_between(ns, [m-s for m,s in zip(means,stds)],
                                    [m+s for m,s in zip(means,stds)],
                                alpha=0.15, color=c)
            ax.set_xscale("log")
            ax.set_xlabel("Labeled passages"); ax.set_ylabel("Accuracy")
            ax.set_title(f"{key}\n{data['label_names']}", fontsize=10)
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(fig_dir / f"probe_{key}.pdf", bbox_inches="tight")
            plt.close(fig)
    except Exception as e:
        print(f"  [WARN] Figures: {e}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="./hierarchical_sae_outputs")
    p.add_argument("--results_dir", default="./results/rw_rq2")
    p.add_argument("--models", nargs="+", default=["pythia1b4", "gemma2_9b"])
    p.add_argument("--datasets", nargs="+", default=["amps", "arxiv", "wikipedia"])
    p.add_argument("--arxiv_local_path",
                   default="your_path_to_arxiv_metadata.json",
                   help="Path to local arXiv JSONL file (same as pipeline config)")
    p.add_argument("--model_order", type=int, default=10)
    p.add_argument("--lambda_sp", type=float, default=1e-4)
    p.add_argument("--epoch", type=int, default=30)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--z_per_layer", type=int, default=64)
    p.add_argument("--n_seeds", type=int, default=5)
    p.add_argument("--min_freq", type=int, default=50)
    p.add_argument("--diagnose_labels", action="store_true",
                   help="Print label samples and exit without running probe.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.diagnose_labels:
        print("=== LABEL DIAGNOSTIC ===")
        for mtag in args.models:
            for dtag in args.datasets:
                bp = bnd_path(args.output_dir, mtag, dtag)
                if not bp.exists():
                    print(f"  [MISSING boundaries] {bp}"); continue
                bounds = np.load(bp)
                n_passages = len(bounds) - 1
                print(f"\n{mtag} x {dtag}: {n_passages} passages")
                labels = load_labels_for_dataset(
                    dtag, min(n_passages, 500), args.output_dir,
                    arxiv_local_path=args.arxiv_local_path,
                )
                labels_200 = [l for l in labels[:200] if l is not None]
                print(f"  Distribution (first 200): "
                      f"{dict(Counter(labels_200).most_common(10))}")
    else:
        run_rq2(args)