#!/bin/bash
#SBATCH --job-name=cltd_data_gen
#SBATCH --partition=preempt
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --array=0-0   # 13 tasks = 0 to 12, number of max concurrent task = 4
#SBATCH --mail-type=END
#SBATCH --requeue

# Load your environment if needed
CONDA_BASE="$HOME/miniconda3"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate cltd

# Define an array of all your commands
commands=(
"python train.py --data_path data/uniform_3_same.npz --nlayer 3"
)

# Get the command for this job
CMD="${commands[$SLURM_ARRAY_TASK_ID]}"

echo "Running task $SLURM_ARRAY_TASK_ID: $CMD"
srun --ntasks=1 --exclusive $CMD