# Synthetic adaptation — selected replay-free implementation

The sibling synthetic_adaptation_archived directory preserves the previous
implementation intact. This directory is the active replacement.

## Restored base components

- more_core.py — original synthetic generator, coordinate basis, MoRe losses,
  and evaluation utilities.
- more_baselines.py, more_sweep.py, merge_sweep.py — previous baseline and
  sweep components retained for compatibility.

## Selected gate and runners

- replayfree_gate.py — model-state-only gate. Stage 2 accepts a saved
  ReplayFreeOldModelState plus new observations, never old samples or labels.
- run_replayfree_c1_c7.py — C1--C7 replay-free validation.
- run_replayfree_c11_c17.py — three-layer/five-component replay-free
  validation.
- run_c11_c17.py — shared three-layer data generator and block-coordinate
  model used by the replay-free C11--C17 runner.

rank_aware_gate.py is retained only for shared gate data structures and
backward compatibility. The selected method is replayfree_gate.py.

Run from the project root:

    python synthetic/synthetic_adaptation/run_replayfree_c1_c7.py --device cpu
    python synthetic/synthetic_adaptation/run_replayfree_c11_c17.py --device cpu
