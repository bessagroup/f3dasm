#!/bin/bash
#SBATCH --job-name=evaluate_golden
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --partition=compute
#SBATCH --account=proj123
#SBATCH --output=/golden/GOLDEN/logs/evaluate_%A_%a.out

module load python/3.11

export MY_VAR="value"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

python -m f3dasm.pipeline.run_step \
  --step=evaluate \
  --job-dir=/golden/GOLDEN \
  --project-dir=eval \
  --iteration=0 \
  --job-number=$SLURM_ARRAY_TASK_ID
