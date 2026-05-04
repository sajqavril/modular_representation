#!/bin/bash
#SBATCH --job-name=hier-sae
#SBATCH --partition=general
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --gres=gpu:A6000:1
#SBATCH --array=0-23
#SBATCH --mail-type=END,FAIL
#SBATCH --requeue
#SBATCH --output=logs/hier_sae_%A_%a.out
#SBATCH --error=logs/hier_sae_%A_%a.err

# ── environment ─────────────────────────────────────────────────────────────
set -euo pipefail

mkdir -p logs


OUTPUT_DIR=/data/user_data/jiaqisun/cltd_realworld/interpretability/hier_sae_outputs

# ── one-time extraction guard ────────────────────────────────────────────────
# Extraction is shared across all array tasks — only run it from task 0,
# and all other tasks wait until the activation files exist.
# (Uses a lockfile so parallel array start doesn't race.)
LOCK_FILE=${OUTPUT_DIR}/.extract_done

if [ "${SLURM_ARRAY_TASK_ID}" -eq 0 ]; then
    if [ ! -f "${LOCK_FILE}" ]; then
        echo "[task 0] Running extraction stage ..."
        python run_pipeline.py \
            --stage extract \
            --output_dir "${OUTPUT_DIR}"
        touch "${LOCK_FILE}"
        echo "[task 0] Extraction complete."
    else
        echo "[task 0] Extraction already done, skipping."
    fi
else
    echo "[task ${SLURM_ARRAY_TASK_ID}] Waiting for extraction ..."
    while [ ! -f "${LOCK_FILE}" ]; do
        sleep 30
    done
    echo "[task ${SLURM_ARRAY_TASK_ID}] Extraction ready, starting training."
fi

# ── training ─────────────────────────────────────────────────────────────────
echo "Starting training: task_id=${SLURM_ARRAY_TASK_ID}"
python run_pipeline.py \
    --stage train \
    --task_id "${SLURM_ARRAY_TASK_ID}" \
    --output_dir "${OUTPUT_DIR}"

echo "Job complete: task_id=${SLURM_ARRAY_TASK_ID}"