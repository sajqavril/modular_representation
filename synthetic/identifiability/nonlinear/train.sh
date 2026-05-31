#!/usr/bin/env bash
set -euo pipefail

EXP=${1:-I}
SEED=${2:-42}
CONFIG="configs/${EXP}.yaml"

SAMPLES=${SAMPLES:-10000}
LENGTH=${LENGTH:-5}
COND_SAMPLES=${COND_SAMPLES:-10000}
HIST=${HIST:-0}
INST=${INST:-0}

if [ ! -f "${CONFIG}" ]; then
  echo "Missing config: ${CONFIG}" >&2
  exit 1
fi

gen_cmd=(
  python -u tools/gen_data.py
  --config "${CONFIG}"
  --seed "${SEED}"
  --hist "${HIST}"
  --batch-size "${SAMPLES}"
  --length "${LENGTH}"
  --cond-samples "${COND_SAMPLES}"
)

if [ "${INST}" = "1" ] || [ "${INST}" = "true" ]; then
  gen_cmd+=(--inst)
fi

echo "Generating dataset from ${CONFIG}"
"${gen_cmd[@]}"

echo "Starting training: exp=${EXP} seed=${SEED}"
python -u train_instantaneous2.py -e "${EXP}" -s "${SEED}"
