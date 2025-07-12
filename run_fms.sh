#!/bin/bash
#SBATCH --account=bebv-dtai-gh
#SBATCH --job-name=fms
#SBATCH --output=slurm_fms_%j.out
#SBATCH --partition=ghx4
##SBATCH --reservation=affinity1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=0
#SBATCH --time=08:00:00

set -x

cd $SLURM_SUBMIT_DIR
echo "RUNANDTIME_START $(date +%s)"
./hf_batch.py --prompt_tokens 32 --max_new_tokens 32 --batch_size 128 --n_batch 4
echo "RUNANDTIME_STOP $(date +%s)"

