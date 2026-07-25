#!/bin/bash
#SBATCH --partition=general
#SBATCH --job-name=more_baselines
#SBATCH --output=logs/baselines_%A_%a.out
#SBATCH --error=logs/baselines_%A_%a.err
#SBATCH --array=0-2          # 3 seeds
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

mkdir -p logs

SEED=${SLURM_ARRAY_TASK_ID}

python more_baselines.py \
    --out_dir results/baselines \
    --T 50000 \
    --obs_dim 8 \
    --noise_scale 0.1 \
    --seeds ${SEED} \
    --device cuda \
    --old_epochs 3000 \
    --epochs 5000 \
    --lr 0.005 \
    --tau_start 2.0 \
    --tau_end 0.3 \
    --warmup_frac 0.3 \
    --w_data 1.0 \
    --w_align 0.01 \
    --w_sp_old 0.01 \
    --w_sp_new 0.05 \
    --w_B_sp 0.01 \
    --w_gate_binary 0.005 \
    --analytic_gate_C 3.0 \
    --gate_lr_factor 0.1 \
    --cases 0 1 2 3 4 5 6

echo "Done seed=${SEED}"
