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

# set -x

cd $SLURM_SUBMIT_DIR
source fms_env

# FMS_ARGS=(
#   --prompt_tokens 32
#   --max_new_tokens 32
#   --batch_size 128
#   --n_batch 4
# )

# FMS_SCRIPT=hf_batch.py

TORCHRUN_ARGS=(
  --nproc_per_node=1
  scripts/inference.py
  --architecture hf_pretrained
  --variant meta-llama/Llama-3.1-8B
  --tokenizer meta-llama/Llama-3.1-8B
)

echo "FMS_START $(date +%s)"
torchrun ${TORCHRUN_ARGS[@]}
echo "FMS_STOP $(date +%s)"

