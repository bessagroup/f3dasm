#!/bin/bash
#SBATCH --job-name=make_golden
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --partition=compute
#SBATCH --account=proj123
#SBATCH --output=/golden/GOLDEN/logs/make_%j.out

module load python/3.11

export MY_VAR="value"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

python -m f3dasm.pipeline.run_step \
  --step=make \
  --job-dir=/golden/GOLDEN \
  --project-dir=. \
  --iteration=0
