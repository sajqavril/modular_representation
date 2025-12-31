#!/bin/bash
#SBATCH --job-name=cltd_data_gen
#SBATCH --partition=preempt
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=30G
#SBATCH --gres=gpu:1
#SBATCH --array=0-5   # 13 tasks = 0 to 12, number of max concurrent task = 4
#SBATCH --mail-type=END
#SBATCH --requeue

# Load your environment if needed
# module load python/3.9

# Define an array of all your commands
commands=(
"python cltd_data_generator.py --nlags 2 --nlatents 4 --length 10 --seed 1"
"python cltd_data_generator.py --nlags 2 --nlatents 4 --length 10 --seed 2"
"python cltd_data_generator.py --nlags 2 --nlatents 4 --length 10 --seed 3"
"python cltd_data_generator.py --nlags 3 --nlatents 4 --length 10 --seed 1"
"python cltd_data_generator.py --nlags 3 --nlatents 4 --length 10 --seed 2"
"python cltd_data_generator.py --nlags 3 --nlatents 4 --length 10 --seed 3"
)

# Get the command for this job
CMD="${commands[$SLURM_ARRAY_TASK_ID]}"

echo "Running task $SLURM_ARRAY_TASK_ID: $CMD"
srun --ntasks=1 --exclusive $CMD