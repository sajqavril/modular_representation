# Synthetic adaptation — updated MoRe implementation

The sibling synthetic_adaptation_archived directory preserves the previous
implementation intact. This directory is the active replacement.

## Restored base components

- more_core.py — original synthetic generator, coordinate basis, MoRe losses,
  and evaluation utilities.
- more_baselines.py, more_sweep.py, merge_sweep.py — previous baseline and
  sweep components retained for compatibility.

## Updated adaptation and runners

The active implementation is the replay-free updated MoRe version:

- `more_adaptation.py` — public adaptation API: saved model state,
  target-span gate, rank-derived expansion count, and target-only alignment.
- `run_adaptation_c1_c7.py` — C1--C7 runner.
- `run_adaptation_c11_c17.py` — three-layer C11--C17 runner.

At adaptation, the public API accepts only a saved `ProfileOldState` and new
observations. It does not accept old samples, latent labels, scenario masks,
or a known number of modules to add. The gate uses stored decoder directions
and target-span membership; it does not use labels or old-data statistics.

Run the selected implementation from the project root:

    conda run -n cltd python synthetic/synthetic_adaptation/run_adaptation_c1_c7.py --device cpu
    conda run -n cltd python synthetic/synthetic_adaptation/run_adaptation_c11_c17.py --device cpu

The three choices that were flags in the historical gate-inspection runner are
now the default behavior in `run_adaptation_c1_c7.py`: target `t=0` source initialization,
frozen profile-initialized new encoders, and target-only old-coordinate
alignment.

## Legacy replay-free gate and runners

- replayfree_gate.py — model-state-only gate. Stage 2 accepts a saved
  ReplayFreeOldModelState plus new observations, never old samples or labels.
- run_replayfree_c1_c7.py — C1--C7 replay-free validation.
- run_replayfree_c11_c17.py — three-layer/five-component replay-free
  validation.
- run_c11_c17.py — shared three-layer data generator and block-coordinate
  model used by the replay-free C11--C17 runner.

`replayfree_gate.py` and `rank_aware_gate.py` are retained for backward
compatibility and ablations only; they are not the selected profile method.

Run from the project root:

    python synthetic/synthetic_adaptation/run_replayfree_c1_c7.py --device cpu
    python synthetic/synthetic_adaptation/run_replayfree_c11_c17.py --device cpu
