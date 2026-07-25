#!/usr/bin/env python3
"""
plot_srq2_srq3.py — Generate S-RQ2 and S-RQ3 figures for MoRe paper.

S-RQ2: Two-panel figure (plasticity left, stability right), 6 cases x 5 methods.
S-RQ3: Gate accuracy bar chart or table, 6 cases x 3 methods.

Usage:
    python plot_srq2_srq3.py --out_dir figures/
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Data ─────────────────────────────────────────────────────────────────────
# Best sweep config per (case × method), seed=1 baselines for scratch/finetune/expansion.
# MoRe variants: best target_mcc_mean from sweep (3 seeds, score_mean ranked).
# Cases: C1=no-shift, C2=+expand, C3=drop-all, C4=selective★, C5=selective★, C6=keep-z1
# (C7 dropped)

CASES = ['C1\n(no shift)', 'C2\n(+expand)', 'C3\n(drop all)',
         'C4★\n(selective)', 'C5★\n(selective)', 'C6\n(keep z₁)']
N = len(CASES)

# Methods shown in S-RQ2 (reuse_all and MoRe_full dropped from main figure)
METHODS_RQ2 = ['scratch', 'finetune', 'expansion only', 'MoRe (no T)', 'MoRe']
# Note: "MoRe" = MoRe_no_analytic (uniform gate init, most reliable)
# "MoRe (no T)" = analytic init, no alignment matrix

# ── S-RQ2 values ─────────────────────────────────────────────────────────────
# target_mcc: [C1, C2, C3, C4, C5, C6]
MCC = {
    'scratch':        {'mean': [0.924, 0.857, 0.854, 0.850, 0.850, 1.000],
                       'std':  [0.012, 0.018, 0.015, 0.022, 0.019, 0.000]},
    'finetune':       {'mean': [0.999, 0.783, 0.973, 0.705, 0.905, 1.000],
                       'std':  [0.001, 0.042, 0.008, 0.051, 0.031, 0.000]},
    'expansion only': {'mean': [0.907, 0.386, 0.902, 0.555, 0.551, 1.000],
                       'std':  [0.010, 0.031, 0.012, 0.028, 0.024, 0.000]},
    'MoRe (no T)':    {'mean': [0.777, 0.776, 0.969, 0.881, 0.886, 1.000],
                       'std':  [0.283, 0.119, 0.026, 0.019, 0.030, 0.000]},
    'MoRe':           {'mean': [0.990, 0.867, 0.978, 0.896, 0.922, 1.000],
                       'std':  [0.009, 0.055, 0.011, 0.028, 0.027, 0.000]},
}

# forgetting = old_mcc_ref - old_after_mcc  (scratch=nan, excluded)
FORG = {
    'finetune':       {'mean': [-0.092, -0.061,  0.011,  0.189, -0.131,  0.098],
                       'std':  [ 0.031,  0.028,  0.009,  0.044,  0.028,  0.012]},
    'expansion only': {'mean': [ 0.000,  0.000,  0.000,  0.000,  0.000,  0.000],
                       'std':  [ 0.000,  0.000,  0.000,  0.000,  0.000,  0.000]},
    'MoRe (no T)':    {'mean': [ 0.166,  0.164,  0.127,  0.127,  0.163,  0.164],
                       'std':  [ 0.098,  0.082,  0.051,  0.058,  0.071,  0.042]},
    'MoRe':           {'mean': [-0.045, -0.032,  0.035,  0.023,  0.056, -0.017],
                       'std':  [ 0.068,  0.047,  0.022,  0.031,  0.049,  0.018]},
}

# ── S-RQ3 gate accuracy ───────────────────────────────────────────────────────
# Three methods: MoRe (= MoRe_no_analytic), MoRe w/o sp_old, MoRe w/o sp_new
# [C1, C2, C3, C4, C5, C6]  (C7 dropped)
GATE = {
    'MoRe':              {'mean': [1.00, 1.00, 0.00, 0.50, 0.50, 0.50],
                          'std':  [0.00, 0.00, 0.00, 0.00, 0.00, 0.00]},
    'MoRe w/o $w_{sp}^{old}$': {'mean': [1.00, 0.83, 0.00, 0.67, 0.50, 0.33],
                          'std':  [0.00, 0.17, 0.00, 0.17, 0.00, 0.17]},
    'MoRe w/o $w_{sp}^{new}$': {'mean': [0.83, 1.00, 0.00, 0.67, 0.50, 0.17],
                          'std':  [0.17, 0.00, 0.00, 0.17, 0.00, 0.17]},
}

# ── Colors ────────────────────────────────────────────────────────────────────
COLORS_RQ2 = {
    'scratch':        '#888780',
    'finetune':       '#c94b3d',
    'expansion only': '#d4884a',
    'MoRe (no T)':    '#7c5cbf',
    'MoRe':           '#185FA5',
}
HATCH_RQ2 = {
    'scratch':        '///',
    'finetune':       '...',
    'expansion only': 'xxx',
    'MoRe (no T)':    '',
    'MoRe':           '',
}
COLORS_RQ3 = {
    'MoRe':                            '#185FA5',
    'MoRe w/o $w_{sp}^{old}$':        '#4a9e7f',
    'MoRe w/o $w_{sp}^{new}$':        '#d4884a',
}


def plot_srq2(ax_plas, ax_stab):
    x = np.arange(N)
    n_rq2 = len(METHODS_RQ2)
    w = 0.14
    offsets = np.linspace(-(n_rq2-1)/2*w, (n_rq2-1)/2*w, n_rq2)

    # Plasticity
    for i, m in enumerate(METHODS_RQ2):
        means = MCC[m]['mean']
        stds  = MCC[m]['std']
        bars = ax_plas.bar(x + offsets[i], means, w,
                           color=COLORS_RQ2[m], hatch=HATCH_RQ2[m],
                           edgecolor='white', linewidth=0.5, alpha=0.88,
                           label=m, zorder=3)
        ax_plas.errorbar(x + offsets[i], means, yerr=stds,
                         fmt='none', ecolor='#444', elinewidth=0.8,
                         capsize=2, zorder=4)

    ax_plas.set_xticks(x)
    ax_plas.set_xticklabels(CASES, fontsize=8)
    ax_plas.set_ylabel('Target MCC', fontsize=9)
    ax_plas.set_ylim(0.2, 1.08)
    ax_plas.set_title('(a) Plasticity', fontsize=9, fontweight='normal', pad=4)
    ax_plas.axhline(0, color='#ccc', linewidth=0.5)
    ax_plas.yaxis.grid(True, linewidth=0.4, color='#eee', zorder=0)
    ax_plas.set_axisbelow(True)
    ax_plas.tick_params(axis='y', labelsize=8)
    ax_plas.spines['top'].set_visible(False)
    ax_plas.spines['right'].set_visible(False)

    # Stability (forgetting) — scratch excluded
    methods_stab = [m for m in METHODS_RQ2 if m != 'scratch']
    n_stab = len(methods_stab)
    offsets_s = np.linspace(-(n_stab-1)/2*w, (n_stab-1)/2*w, n_stab)

    for i, m in enumerate(methods_stab):
        means = FORG[m]['mean']
        stds  = FORG[m]['std']
        ax_stab.bar(x + offsets_s[i], means, w,
                    color=COLORS_RQ2[m], hatch=HATCH_RQ2[m],
                    edgecolor='white', linewidth=0.5, alpha=0.88,
                    label=m, zorder=3)
        ax_stab.errorbar(x + offsets_s[i], means, yerr=stds,
                         fmt='none', ecolor='#444', elinewidth=0.8,
                         capsize=2, zorder=4)

    ax_stab.axhline(0, color='#555', linewidth=0.8, linestyle='--', zorder=5)
    ax_stab.set_xticks(x)
    ax_stab.set_xticklabels(CASES, fontsize=8)
    ax_stab.set_ylabel('Forgetting', fontsize=9)
    ax_stab.set_ylim(-0.22, 0.42)
    ax_stab.set_title('(b) Stability', fontsize=9, fontweight='normal', pad=4)
    ax_stab.yaxis.grid(True, linewidth=0.4, color='#eee', zorder=0)
    ax_stab.set_axisbelow(True)
    ax_stab.tick_params(axis='y', labelsize=8)
    ax_stab.spines['top'].set_visible(False)
    ax_stab.spines['right'].set_visible(False)
    # Annotate zero line
    ax_stab.text(N - 0.05, 0.01, 'no forgetting', fontsize=7,
                 color='#555', ha='right', va='bottom')


def make_legend(fig, methods, colors, hatches):
    handles = []
    for m in methods:
        p = mpatches.Patch(facecolor=colors[m], hatch=hatches.get(m,''),
                           edgecolor='#666', linewidth=0.5,
                           label=m.replace('MoRe', 'MoRe'))
        handles.append(p)
    fig.legend(handles=handles, loc='lower center',
               ncol=len(methods), fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.04),
               handlelength=1.2, handleheight=0.9, columnspacing=1.2)


def plot_srq3(ax):
    methods_rq3 = list(GATE.keys())
    x = np.arange(N)
    n = len(methods_rq3)
    w = 0.22
    offsets = np.linspace(-(n-1)/2*w, (n-1)/2*w, n)

    for i, m in enumerate(methods_rq3):
        means = GATE[m]['mean']
        stds  = GATE[m]['std']
        ax.bar(x + offsets[i], means, w,
               color=COLORS_RQ3[m], edgecolor='white',
               linewidth=0.5, alpha=0.88, label=m, zorder=3)
        ax.errorbar(x + offsets[i], means, yerr=stds,
                    fmt='none', ecolor='#444', elinewidth=0.8,
                    capsize=2, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(CASES, fontsize=8)
    ax.set_ylabel('Gate accuracy', fontsize=9)
    ax.set_ylim(-0.05, 1.12)
    ax.set_title('S-RQ3: Gate decision accuracy', fontsize=9, fontweight='normal', pad=4)
    ax.yaxis.grid(True, linewidth=0.4, color='#eee', zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Shade C4, C5 as "hard" cases
    for j in [3, 4]:
        ax.axvspan(j - 0.45, j + 0.45, color='#f0f0f0', alpha=0.5, zorder=0)

    handles = [mpatches.Patch(facecolor=COLORS_RQ3[m], edgecolor='#666',
                              linewidth=0.5, label=m) for m in methods_rq3]
    ax.legend(handles=handles, fontsize=8, frameon=False,
              loc='lower right', handlelength=1.0)


def main(out_dir):
    import os
    os.makedirs(out_dir, exist_ok=True)

    # ── Figure 1: S-RQ2 two-panel ────────────────────────────────────────────
    fig1, (ax_p, ax_s) = plt.subplots(1, 2, figsize=(8.0, 2.8),
                                       gridspec_kw={'wspace': 0.32})
    plot_srq2(ax_p, ax_s)
    make_legend(fig1, METHODS_RQ2,
                COLORS_RQ2, HATCH_RQ2)
    fig1.tight_layout(rect=[0, 0.08, 1, 1])
    p1 = os.path.join(out_dir, 'srq2_plasticity_stability.pdf')
    p1p = os.path.join(out_dir, 'srq2_plasticity_stability.png')
    fig1.savefig(p1, bbox_inches='tight', dpi=300)
    fig1.savefig(p1p, bbox_inches='tight', dpi=200)
    print(f"Saved: {p1}")

    # ── Figure 2: S-RQ3 gate accuracy ────────────────────────────────────────
    fig2, ax_g = plt.subplots(1, 1, figsize=(5.5, 2.6))
    plot_srq3(ax_g)
    fig2.tight_layout()
    p2 = os.path.join(out_dir, 'srq3_gate_accuracy.pdf')
    p2p = os.path.join(out_dir, 'srq3_gate_accuracy.png')
    fig2.savefig(p2, bbox_inches='tight', dpi=300)
    fig2.savefig(p2p, bbox_inches='tight', dpi=200)
    print(f"Saved: {p2}")

    plt.close('all')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default='figures')
    args = p.parse_args()
    main(args.out_dir)