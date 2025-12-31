#!/bin/bash
#SBATCH --job-name=cltd_poc
#SBATCH --partition=preempt
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=30G
#SBATCH --gres=gpu:1
#SBATCH --array=0-0   # 13 tasks = 0 to 12, number of max concurrent task = 4
#SBATCH --mail-type=END
#SBATCH --requeue

# Load your environment if needed
# module load python/3.9

# Define an array of all your commands
commands=(
"python -u main_poc.py --runs 5"
)

# Get the command for this job
CMD="${commands[$SLURM_ARRAY_TASK_ID]}"

echo "Running task $SLURM_ARRAY_TASK_ID: $CMD"
srun --ntasks=1 --exclusive $CMD