#!/bin/bash
#SBATCH --job-name=loop2_run_golden
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --partition=compute
#SBATCH --account=proj123
#SBATCH --output=/golden/GOLDEN/logs/loop2_run_%A_%a.out

module load python/3.11

export MY_VAR="value"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

python -m f3dasm.pipeline.run_step \
  --step=run \
  --job-dir=/golden/GOLDEN \
  --project-dir=. \
  --iteration=$F3DASM_ITERATION \
  --job-number=$SLURM_ARRAY_TASK_ID
