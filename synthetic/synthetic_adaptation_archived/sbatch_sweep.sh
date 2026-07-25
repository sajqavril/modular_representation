#!/bin/bash
#SBATCH --partition=general
#SBATCH --job-name=more_sweep
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err
#SBATCH --array=0-6            # 7 cases, one job per case; seeds run inside each job
#SBATCH --time=00:25:00        # 8 configs x 3 seeds x 7 methods x ~10s + old_model = ~18 min
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
 
mkdir -p logs results/sweep
 
CASE=${SLURM_ARRAY_TASK_ID}
echo "Task: case=${CASE}, seeds=0,1,2"
 
python more_sweep.py \
    --out_dir results/sweep \
    --T 50000 --obs_dim 8 --noise_scale 0.1 \
    --seeds 0 1 2 --device cuda \
    --old_epochs 1500 --epochs 2000 --lr 0.005 \
    --tau_start 2.0 --w_B_sp 0.01 --gate_lr_factor 0.1 \
    --w_sp_old_grid      "0.005,0.01,0.05" \
    --w_sp_new_grid      "0.001,0.01,0.05" \
    --w_align_grid       "0.005,0.02" \
    --w_gate_binary_grid "0.001,0.01" \
    --analytic_gate_C_grid "2.0,5.0" \
    --warmup_frac_grid   "0.2" \
    --tau_end_grid       "0.2,0.5" \
    --noise_floor_grid   "0.3,0.5" \
    --temperature_grid   "0.3" \
    --max_configs 8 --random_search \
    --cases ${CASE}
 
echo "Done case=${CASE}"