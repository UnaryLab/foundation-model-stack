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
# torchrun --nproc_per_node=1 scripts/inference.py --variant=7b --model_source=meta --model_path=~/.llama/checkpoints/Llama-2-7b --tokenizer=~/.llama/checkpoints/Llama-2-7b/tokenizer.model # --distributed
torchrun --nproc_per_node=1 hf_batch.py
echo "RUNANDTIME_STOP $(date +%s)"

