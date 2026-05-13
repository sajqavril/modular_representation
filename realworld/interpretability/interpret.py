"""
Hierarchical SAE — Interpretation Script
=========================================
Runs analysis tasks across all 6 (model_order, lambda_sp) combos
and all 3 datasets for pythia1b4 epoch030 checkpoints.

Tasks
-----
1. Gini concentration           — how narrowly each concept fires across labels
2. Cross-level B-matrix links   — which L0 concepts causally predict L3 concepts

Outputs (saved to output_dir/interpretation/)
---------------------------------------------
  gini_{ds}_{order}_{lsp}.pdf
  crosslinks_{ds}_{order}_{lsp}.json
  crosslinks_{ds}_{order}_{lsp}.pdf
  block_strengths_{ds}_{order}_{lsp}.json
  summary_table.csv              — all Gini scores across all combos
"""

import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import IncrementalPCA
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════════════
# CONFIG — edit these paths to match your cluster
# ════════════════════════════════════════════════════════════════════════════

OUTPUT_DIR   = Path("/data/user_data/jiaqisun/cltd_realworld/interpretability/hier_sae_outputs")
INTERP_DIR   = OUTPUT_DIR / "interpretation"
MODEL_TAG    = "pythia1b4"
EPOCH        = 30
N_LAYERS     = 4
Z_PER_LAYER  = 64
Z_DIM        = N_LAYERS * Z_PER_LAYER   # 256

DATASETS = ["arxiv", "amps", "wikipedia"]

MODEL_ORDERS = [10, 50, 100]
LAMBDA_SPS   = [1e-4, 1e-5]

# arXiv category → abstraction tier (for labeling)
ARXIV_TIER = {
    "math.LO": 0, "math.CT": 0, "math.LO": 0, "cs.LO": 0,
    "math.CA": 1, "math.GR": 1, "math.AG": 1, "math.NT": 1,
    "cs.CC":   1, "math.FA": 1,
    "cs.LG":   2, "cs.IT":   2, "stat.TH": 2, "cs.AI":  2,
    "stat.ML": 2, "cs.CV":   2, "cs.NE":   2,
    "q-bio":   3, "econ":    3, "cond-mat":3, "astro-ph":3,
    "q-fin":   3, "physics": 3,
}

TIER_NAMES = {0: "L0 Fundamental", 1: "L1 Mid", 2: "L2 Bridge", 3: "L3 Specific"}


# ════════════════════════════════════════════════════════════════════════════
# PATH HELPERS
# ════════════════════════════════════════════════════════════════════════════

def ckpt_path(ds_tag, model_order, lambda_sp):
    lsp_str = f"{lambda_sp:.0e}".replace("e-0", "e-")
    d = (OUTPUT_DIR / "checkpoints" /
         f"{MODEL_TAG}__{ds_tag}__ord{model_order:03d}__lsp{lsp_str}")
    return d / f"epoch{EPOCH:03d}.pt"

def act_file(ds_tag):
    return OUTPUT_DIR / "activations" / f"{MODEL_TAG}__{ds_tag}.npy"

def bnd_file(ds_tag):
    return OUTPUT_DIR / "activations" / f"{MODEL_TAG}__{ds_tag}.boundaries.npy"

def cat_file(ds_tag):
    return OUTPUT_DIR / "activations" / f"{MODEL_TAG}__{ds_tag}.categories.npy"

def combo_tag(model_order, lambda_sp):
    lsp_str = f"{lambda_sp:.0e}".replace("e-0", "e-")
    return f"ord{model_order:03d}_lsp{lsp_str}"


# ════════════════════════════════════════════════════════════════════════════
# LOAD HELPERS
# ════════════════════════════════════════════════════════════════════════════

def load_checkpoint(ds_tag, model_order, lambda_sp):
    p = ckpt_path(ds_tag, model_order, lambda_sp)
    if not p.exists():
        print(f"  [MISSING] {p.name}")
        return None
    ckpt = torch.load(p, map_location="cpu")
    g    = ckpt["model_state"]["g_raw"]          # [256, 256]
    f    = torch.linalg.inv(g)                   # demixing matrix
    # collect all B matrices
    Bs = []
    k = 0
    while f"B_list.{k}" in ckpt["model_state"]:
        Bs.append(ckpt["model_state"][f"B_list.{k}"])
        k += 1
    mask = ckpt["model_state"]["mask"]           # [256, 256]
    losses = ckpt.get("losses", {})
    return {"g": g, "f": f, "Bs": Bs, "mask": mask, "losses": losses}


def load_dataset(ds_tag):
    """Load activations, boundaries, categories, raw texts. Returns dict or None."""
    ap = act_file(ds_tag)
    bp = bnd_file(ds_tag)
    cp = cat_file(ds_tag)
    if not ap.exists():
        print(f"  [MISSING activations] {ap.name}")
        return None
    acts       = np.load(ap, mmap_mode="r")
    boundaries = np.load(bp)
    categories = (np.load(cp, allow_pickle=True)
                  if cp.exists() else
                  np.array([""] * (len(boundaries)-1)))
    n_passages = len(boundaries) - 1
    texts  = [""] * n_passages
    titles = [""] * n_passages

    if ds_tag == "amps":
        print(f"  Loading AMPS texts for labeling ({n_passages:,} passages)...")
        try:
            from datasets import load_dataset as hf_load
            ds = hf_load("sarahpann/AMPS", split="train",
                         streaming=True, trust_remote_code=True)
            idx = 0
            for row in ds:
                if idx >= n_passages: break
                p = str(row.get("problem", "")).strip()
                s = str(row.get("step_by_step", "")).strip()
                texts[idx] = (p + " " + s).strip()
                idx += 1
        except Exception as e:
            print(f"  [WARN] Could not load AMPS texts: {e}")

    elif ds_tag == "wikipedia":
        print(f"  Loading Wikipedia texts for labeling ({n_passages:,} passages)...")
        try:
            from datasets import load_dataset as hf_load
            ds = hf_load("wikimedia/wikipedia", "20231101.en",
                         split="train", streaming=True, trust_remote_code=True)
            idx = 0
            for row in ds:
                if idx >= n_passages: break
                titles[idx] = str(row.get("title", "")).strip()
                texts[idx]  = str(row.get("text", "")).strip()
                idx += 1
        except Exception as e:
            print(f"  [WARN] Could not load Wikipedia texts: {e}")

    return {
        "acts":       acts,
        "boundaries": boundaries,
        "categories": categories,
        "texts":      texts,
        "titles":     titles,
    }


def load_pca(model_tag):
    p = OUTPUT_DIR / "pca" / f"{model_tag}__pca.npz"
    if not p.exists():
        raise FileNotFoundError(f"PCA not found: {p}")
    data = np.load(p)
    pca  = IncrementalPCA(n_components=256)
    pca.components_               = data["components"]
    pca.mean_                     = data["mean"]
    pca.explained_variance_       = data["explained_variance"]
    pca.explained_variance_ratio_ = data["explained_variance_ratio"]
    return pca


def encode_z(acts, f_matrix, pca):
    """PCA compress then encode into concept space.
    acts     : [N_tokens, d_model]  float16
    f_matrix : [z_dim, z_dim]       demixing matrix
    pca      : fitted IncrementalPCA
    Returns  : [N_tokens, z_dim]    float32
    """
    N   = acts.shape[0]
    f   = f_matrix.float().numpy()
    out = np.empty((N, 256), dtype=np.float32)
    bs  = 4096
    for i in range(0, N, bs):
        chunk        = acts[i: i+bs].astype(np.float32)
        chunk_pca    = pca.transform(chunk).astype(np.float32)
        out[i: i+bs] = chunk_pca @ f.T
    return out


# ── per-dataset label functions ───────────────────────────────────────────

# AMPS: keyword → (tier, short label)
AMPS_TIER_KEYWORDS = [
    # check most specific first
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

def amps_label(text):
    t = text.lower()
    for tier, keywords in AMPS_TIER_KEYWORDS:
        for kw in keywords:
            if kw in t:
                return f"L{tier}_{kw.replace(' ', '_')}"
    return "L1_general_math"


def wikipedia_label(text, title=""):
    """Use article word count as specificity proxy + title keywords."""
    n_words = len(text.split())
    # also check title for subject clues
    title_l = title.lower()
    specific_keywords = [
        "theorem", "model", "equation", "algorithm", "formula",
        "hypothesis", "lemma", "conjecture", "proof",
    ]
    general_keywords = [
        "mathematics", "science", "history", "philosophy",
        "logic", "language", "art",
    ]
    if any(kw in title_l for kw in specific_keywords):
        return f"L3_specific ({n_words}w)"
    if any(kw in title_l for kw in general_keywords):
        return f"L0_general ({n_words}w)"
    if n_words < 500:   return f"L0_stub ({n_words}w)"
    if n_words < 2000:  return f"L1_overview ({n_words}w)"
    if n_words < 5000:  return f"L2_detailed ({n_words}w)"
    return                     f"L3_specific ({n_words}w)"


def get_primary_category(cat_str, ds_tag, text="", title=""):
    """Return a short human-readable label for a passage."""
    if ds_tag == "arxiv":
        if not cat_str:
            return "other"
        primary = cat_str.split()[0]
        return primary   # e.g. "math.LO", "q-bio.PE", "econ.GN"
    elif ds_tag == "amps":
        return amps_label(text)
    else:   # wikipedia
        return wikipedia_label(text, title)


# ════════════════════════════════════════════════════════════════════════════
# TASK 1 — GINI CONCENTRATION
# ════════════════════════════════════════════════════════════════════════════

def gini_coefficient(x):
    """
    With only 4 groups, use concentration ratio instead of Gini:
    concentration = std / (mean + eps)
    This is more sensitive to small group differences.
    Normalized to [0, 1] range.
    """
    x = np.abs(x).flatten()
    if x.sum() == 0:
        return 0.
    mean = x.mean()
    std  = x.std()
    # coefficient of variation, clipped to [0,1]
    cv = std / (mean + 1e-8)
    # normalize: cv of uniform dist over n groups ≈ 0,
    # cv of all-one-group ≈ sqrt(n-1)
    n = len(x)
    max_cv = np.sqrt(n - 1) if n > 1 else 1.
    return float(np.clip(cv / max_cv, 0., 1.))


def coarsen_label(label, ds_tag):
    """Map fine-grained labels to 4 coarse tiers for Gini computation."""
    if ds_tag == "arxiv":
        for prefix, tier in ARXIV_TIER.items():
            if label.startswith(prefix):
                return f"tier_{tier}"
        return "tier_2"
    elif ds_tag == "amps":
        if label.startswith("L0"): return "tier_0"
        if label.startswith("L1"): return "tier_1"
        if label.startswith("L2"): return "tier_2"
        if label.startswith("L3"): return "tier_3"
        return "tier_1"
    else:
        if "L0" in label or "stub" in label:     return "tier_0"
        if "L1" in label or "overview" in label: return "tier_1"
        if "L2" in label or "detailed" in label: return "tier_2"
        if "L3" in label or "specific" in label: return "tier_3"
        return "tier_1"

def run_gini(ds_tag, Z, boundaries, categories, ax, texts=None, titles=None,
             n_null=20, rng_seed=42):
    """
    Compute per-concept Gini across category mean activations.
    Adds null control: shuffle tier labels n_null times and compare.
    Plot per-layer Gini distribution as violin + null mean line.
    Returns (concept_gini [z_dim], null_layer_means [n_layers]).
    """
    N_passages = len(boundaries) - 1
    gi_texts  = texts  if texts  is not None else [""] * N_passages
    gi_titles = titles if titles is not None else [""] * N_passages
    raw_labels = [
        get_primary_category(
            categories[i], ds_tag,
            text=gi_texts[i]  if i < len(gi_texts)  else "",
            title=gi_titles[i] if i < len(gi_titles) else "",
        )
        for i in range(N_passages)
    ]

    labels      = [coarsen_label(l, ds_tag) for l in raw_labels]
    unique_cats = sorted(set(labels))

    valid_cats = [c for c in unique_cats if labels.count(c) >= 20]
    if not valid_cats:
        valid_cats = unique_cats

    labels_arr = np.array(labels)

    # ── helper: build cat_matrix and compute per-concept Gini ─────────────
    def compute_gini_for_labels(lab_arr):
        mat = np.zeros((len(valid_cats), Z_DIM), dtype=np.float32)
        for ci, cat in enumerate(valid_cats):
            idxs = np.where(lab_arr == cat)[0]
            if len(idxs) == 0:
                continue
            token_idxs = np.concatenate([
                np.arange(int(boundaries[i]), int(boundaries[i+1]))
                for i in idxs
            ])
            if len(token_idxs) == 0:
                continue
            mat[ci] = np.abs(Z[token_idxs]).mean(axis=0)
        return np.array([gini_coefficient(mat[:, k]) for k in range(Z_DIM)])

    # ── real Gini ──────────────────────────────────────────────────────────
    concept_gini = compute_gini_for_labels(labels_arr)
    layer_ginis  = [
        concept_gini[l*Z_PER_LAYER: (l+1)*Z_PER_LAYER]
        for l in range(N_LAYERS)
    ]

    # ── null control: shuffle labels n_null times ──────────────────────────
    rng = np.random.default_rng(rng_seed)
    null_layer_all = np.zeros((n_null, N_LAYERS), dtype=np.float32)
    for run in range(n_null):
        shuffled = rng.permutation(labels_arr)
        null_gini = compute_gini_for_labels(shuffled)
        for l in range(N_LAYERS):
            null_layer_all[run, l] = null_gini[
                l*Z_PER_LAYER: (l+1)*Z_PER_LAYER
            ].mean()
    null_layer_means = null_layer_all.mean(axis=0)   # [n_layers]
    null_layer_stds  = null_layer_all.std(axis=0)

    # ── print summary ──────────────────────────────────────────────────────
    tier_dist = Counter(labels)
    print(f"    Tier distribution: {dict(sorted(tier_dist.items()))}")
    print(f"    Real  Gini by layer: "
          f"{[round(layer_ginis[l].mean(), 4) for l in range(N_LAYERS)]}")
    print(f"    Null  Gini by layer: "
          f"{[round(float(null_layer_means[l]), 4) for l in range(N_LAYERS)]}")

    # ── violin plot ────────────────────────────────────────────────────────
    parts = ax.violinplot(layer_ginis, positions=range(N_LAYERS),
                          showmedians=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("#9b59b6")
        pc.set_alpha(0.6)

    # overlay null mean ± 1 std as shaded band
    x = np.arange(N_LAYERS)
    ax.plot(x, null_layer_means, color="gray", linestyle="--",
            linewidth=1.0, label="null (shuffled)", zorder=3)
    ax.fill_between(x,
                    null_layer_means - null_layer_stds,
                    null_layer_means + null_layer_stds,
                    color="gray", alpha=0.15)

    ax.set_xticks(range(N_LAYERS))
    ax.set_xticklabels(
        [f"L{l}\nμ={layer_ginis[l].mean():.3f}\n"
         f"null={null_layer_means[l]:.3f}"
         for l in range(N_LAYERS)],
        fontsize=7,
    )
    ax.set_ylabel("Gini (higher = more specific)", fontsize=8)
    ax.set_title(f"Concept concentration — {ds_tag}", fontsize=9)
    ax.set_ylim(0, max(0.3, concept_gini.max() * 1.2))
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.5, alpha=0.4)
    ax.legend(fontsize=7, loc="upper right")

    return concept_gini, null_layer_means


# ════════════════════════════════════════════════════════════════════════════
# TASK 2 — CROSS-LEVEL B-MATRIX LINKS
# ════════════════════════════════════════════════════════════════════════════

def tfidf_label(concept_idx, Z, boundaries, raw_labels, top_k=50):
    """
    Use TF-IDF over category label strings of top-k activating passages
    to produce a keyword label for a concept.
    """
    scores = []
    for i in range(len(boundaries) - 1):
        s, e = int(boundaries[i]), int(boundaries[i+1])
        if e <= s:
            continue
        mean_act = float(np.abs(Z[s:e, concept_idx]).mean())
        scores.append((mean_act, i))
    scores.sort(reverse=True)

    top_labels    = [raw_labels[i] for _, i in scores[:top_k]]
    bottom_labels = [raw_labels[i] for _, i in scores[-top_k:]]

    if not any(top_labels):
        return "unknown"

    try:
        vec  = TfidfVectorizer(max_features=500, stop_words="english",
                               ngram_range=(1, 2), min_df=1)
        all_docs = top_labels + bottom_labels
        X        = vec.fit_transform(all_docs)
        names    = vec.get_feature_names_out()
        top_mean = np.asarray(X[:top_k].mean(axis=0)).flatten()
        bot_mean = np.asarray(X[top_k:].mean(axis=0)).flatten()
        diff     = top_mean - bot_mean
        top_terms = [names[i] for i in np.argsort(diff)[-5:][::-1]]
        return " | ".join(top_terms)
    except Exception:
        return top_labels[0] if top_labels else "unknown"


def run_crosslinks(ds_tag, Z, boundaries, categories, texts, titles,
                   Bs, mask, ax, top_n=10):
    """
    Find top cross-level links for ALL layer pairs (Lsrc→Ltgt where tgt>src).
    Plots L0→L3 bipartite graph on ax.
    Returns (links_L0L3, all_block_strengths).

    all_block_strengths: dict keyed by "L{src}->L{tgt}", each value is a list
    of top-3 link dicts with keys: strength, src_concept, tgt_concept,
    src_label, tgt_label, src_top_cats, tgt_top_cats.
    """
    N_passages = len(boundaries) - 1
    raw_labels = [
        get_primary_category(categories[i], ds_tag,
                             text=texts[i]  if i < len(texts)  else "",
                             title=titles[i] if i < len(titles) else "")
        for i in range(N_passages)
    ]

    # ── aggregate B matrices: mean absolute weight across all lag orders ──
    B_agg = np.zeros((Z_DIM, Z_DIM), dtype=np.float32)
    for B_raw in Bs:
        B_agg += np.abs((B_raw * mask).float().numpy())
    B_agg /= max(len(Bs), 1)

    # ── helper: top-n links from an arbitrary cross-block ────────────────
    def extract_links_from_block(src_l, tgt_l, n=top_n):
        src_start = src_l * Z_PER_LAYER
        tgt_start = tgt_l * Z_PER_LAYER
        block     = B_agg[tgt_start: tgt_start + Z_PER_LAYER,
                          src_start: src_start + Z_PER_LAYER]   # [64, 64]
        flat_idx  = np.argsort(block.ravel())[-n:][::-1]
        result    = []
        for fi in flat_idx:
            tgt_local  = fi // Z_PER_LAYER
            src_local  = fi  % Z_PER_LAYER
            src_global = src_start + src_local
            tgt_global = tgt_start + tgt_local
            strength   = float(block[tgt_local, src_local])

            src_label = tfidf_label(src_global, Z, boundaries, raw_labels)
            tgt_label = tfidf_label(tgt_global, Z, boundaries, raw_labels)

            cat_acts_src = defaultdict(list)
            cat_acts_tgt = defaultdict(list)
            for i in range(N_passages):
                s, e = int(boundaries[i]), int(boundaries[i+1])
                if e <= s:
                    continue
                cat = raw_labels[i]
                cat_acts_src[cat].append(
                    float(np.abs(Z[s:e, src_global]).mean()))
                cat_acts_tgt[cat].append(
                    float(np.abs(Z[s:e, tgt_global]).mean()))

            top_cats_src = sorted(cat_acts_src,
                                  key=lambda c: np.mean(cat_acts_src[c]),
                                  reverse=True)[:3]
            top_cats_tgt = sorted(cat_acts_tgt,
                                  key=lambda c: np.mean(cat_acts_tgt[c]),
                                  reverse=True)[:3]

            result.append({
                "strength":      strength,
                "src_concept":   int(src_global),
                "tgt_concept":   int(tgt_global),
                "src_layer":     src_l,
                "tgt_layer":     tgt_l,
                "src_label":     src_label,
                "tgt_label":     tgt_label,
                "src_top_cats":  top_cats_src,
                "tgt_top_cats":  top_cats_tgt,
            })
        return result, float(block.mean()), float(block.max())

    # ── extract all 6 cross-layer pairs ───────────────────────────────────
    all_block_strengths = {}
    for src_l in range(N_LAYERS):
        for tgt_l in range(src_l + 1, N_LAYERS):
            key = f"L{src_l}->L{tgt_l}"
            links_block, mean_w, max_w = extract_links_from_block(src_l, tgt_l)
            all_block_strengths[key] = {
                "mean_weight": mean_w,
                "max_weight":  max_w,
                "top_links":   links_block,
            }

    # L0→L3 is the primary result for plotting and return
    links = all_block_strengths["L0->L3"]["top_links"]

    # ── print summary of all block mean weights ───────────────────────────
    print(f"    Block mean weights ({ds_tag}):")
    for key, val in all_block_strengths.items():
        print(f"      {key}: mean={val['mean_weight']:.5f}  "
              f"max={val['max_weight']:.4f}")

    # ── bipartite plot (L0→L3 only, as before) ───────────────────────────
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, top_n - 0.5)
    ax.axis("off")
    ax.set_title(f"L0→L3 cross-level links — {ds_tag}", fontsize=9)

    max_strength = links[0]["strength"] if links else 1.

    for rank, link in enumerate(links):
        y    = top_n - 1 - rank
        w    = link["strength"] / max(max_strength, 1e-6)
        src_s = link["src_label"][:25]
        tgt_s = link["tgt_label"][:25]

        ax.text(-0.05, y, f"L0: {src_s}",
                ha="right", va="center", fontsize=6.5,
                bbox=dict(boxstyle="round,pad=0.2",
                          fc="#D6EAF8", ec="#2980B9", lw=0.7))
        ax.text(1.05, y, f"L3: {tgt_s}",
                ha="left", va="center", fontsize=6.5,
                bbox=dict(boxstyle="round,pad=0.2",
                          fc="#FDEDEC", ec="#E74C3C", lw=0.7))
        ax.annotate("", xy=(1.05, y), xytext=(-0.05, y),
                    arrowprops=dict(
                        arrowstyle="->",
                        color=plt.cm.Blues(0.4 + 0.5 * w),
                        lw=0.5 + 2.0 * w,
                    ))
        ax.text(0.5, y + 0.15, f"{link['strength']:.4f}",
                ha="center", va="bottom", fontsize=6, color="gray")

    return links, all_block_strengths


# ════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ════════════════════════════════════════════════════════════════════════════

def main():
    INTERP_DIR.mkdir(parents=True, exist_ok=True)

    # load all datasets once (shared across combos)
    print("Loading datasets...")
    ds_data = {}
    for ds_tag in DATASETS:
        print(f"  {ds_tag}...")
        data = load_dataset(ds_tag)
        if data is None:
            continue
        ds_data[ds_tag] = data

    if not ds_data:
        print("No datasets found. Check OUTPUT_DIR.")
        return

    print("Loading PCA...")
    pca = load_pca(MODEL_TAG)

    # summary table accumulator
    summary_rows = []

    # ── iterate over all 6 combos ─────────────────────────────────────────
    for model_order in MODEL_ORDERS:
        for lambda_sp in LAMBDA_SPS:
            ctag = combo_tag(model_order, lambda_sp)
            print(f"\n{'='*60}")
            print(f"order={model_order}  lsp={lambda_sp:.0e}")
            print(f"{'='*60}")

            # ── Fig 1: Gini (3 datasets side by side) ────────────────────
            fig_gini, axes_gini = plt.subplots(1, 3, figsize=(15, 4))
            fig_gini.suptitle(
                f"Concept Concentration (Gini) — order={model_order} lsp={lambda_sp:.0e}",
                fontsize=11
            )

            # ── Fig 2: cross-links (3 datasets) ──────────────────────────
            fig_cl, axes_cl = plt.subplots(1, 3, figsize=(21, 8))
            fig_cl.suptitle(
                f"L0→L3 Cross-level Links — order={model_order} lsp={lambda_sp:.0e}",
                fontsize=11
            )

            all_links = {}

            for di, ds_tag in enumerate(DATASETS):
                if ds_tag not in ds_data:
                    continue

                print(f"\n  Dataset: {ds_tag}")
                data = ds_data[ds_tag]

                # load checkpoint
                ckpt = load_checkpoint(ds_tag, model_order, lambda_sp)
                if ckpt is None:
                    continue

                # encode activations → concept space
                print(f"    Encoding {data['acts'].shape[0]:,} tokens...")
                Z = encode_z(data["acts"], ckpt["f"], pca)   # [N_tok, 256]

                # ── Task 1: Gini ──────────────────────────────────────────
                print("    Running Gini...")
                concept_gini, null_means = run_gini(
                    ds_tag, Z,
                    data["boundaries"], data["categories"],
                    axes_gini[di],
                    texts=data.get("texts"),
                    titles=data.get("titles"),
                )

                # accumulate summary
                for l in range(N_LAYERS):
                    layer_gini = concept_gini[l*Z_PER_LAYER: (l+1)*Z_PER_LAYER]
                    summary_rows.append({
                        "dataset":     ds_tag,
                        "model_order": model_order,
                        "lambda_sp":   lambda_sp,
                        "layer":       l,
                        "gini_mean":   float(layer_gini.mean()),
                        "gini_std":    float(layer_gini.std()),
                        "gini_max":    float(layer_gini.max()),
                        "null_mean":   float(null_means[l]),
                    })

                # ── Task 2: cross-level links ─────────────────────────────
                print("    Running cross-level links...")
                links, block_strengths = run_crosslinks(
                    ds_tag, Z,
                    data["boundaries"], data["categories"],
                    data.get("texts",  [""]*(len(data["boundaries"])-1)),
                    data.get("titles", [""]*(len(data["boundaries"])-1)),
                    ckpt["Bs"], ckpt["mask"],
                    axes_cl[di],
                    top_n=10,
                )
                all_links[ds_tag] = {
                    "L0->L3_links": links,
                    "block_strengths": block_strengths,
                }

                # print top 3 L0->L3 as before
                print(f"    Top 3 L0→L3 links:")
                for link in links[:3]:
                    print(f"      [{link['strength']:.4f}] "
                        f"L0({link['src_label'][:30]}) "
                        f"→ L3({link['tgt_label'][:30]})")
                    print(f"        src cats: {link['src_top_cats']}")
                    print(f"        tgt cats: {link['tgt_top_cats']}")

                del Z   # free RAM before next dataset

            # save figures
            gi_path = INTERP_DIR / f"gini_{ctag}.pdf"
            fig_gini.savefig(gi_path, bbox_inches="tight", dpi=150)
            plt.close(fig_gini)
            print(f"\n  Saved → {gi_path.name}")

            cl_path = INTERP_DIR / f"crosslinks_{ctag}.pdf"
            fig_cl.savefig(cl_path, bbox_inches="tight", dpi=150)
            plt.close(fig_cl)
            print(f"  Saved → {cl_path.name}")

            # save block strengths as JSON
            blocks_path = INTERP_DIR / f"block_strengths_{ctag}.json"
            with open(blocks_path, "w") as fh:
                json.dump({ds: v["block_strengths"]
                        for ds, v in all_links.items()}, fh, indent=2)
            print(f"  Saved → {blocks_path.name}")

    # ── summary table ─────────────────────────────────────────────────────
    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        csv_path   = INTERP_DIR / "summary_gini.csv"
        df_summary.to_csv(csv_path, index=False)
        print(f"\nSaved summary → {csv_path.name}")

        # print pivot: expected to show gini increasing with layer
        print("\n── Gini mean by dataset × layer (averaged across combos) ──")
        pivot = df_summary.groupby(["dataset", "layer"])["gini_mean"].mean().unstack()
        print(pivot.round(4).to_string())

    print("\n✓ Interpretation complete.")


if __name__ == "__main__":
    main()