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

set -eu

cd $SLURM_SUBMIT_DIR
source fms_env

# FMS_ARGS=(
#   --prompt_tokens 32
#   --max_new_tokens 32
#   --batch_size 128
#   --n_batch 4
# )

# FMS_SCRIPT=hf_batch.py

COUNTERS=0

COUNTERS_ARGS=(
        ncu
        -f
        --target-processes all
        --profile-from-start no
        -o memory
        --section "regex:MemoryWorkloadAnalysis(_Chart|_Tables)?"
        --replay-mode application
)

if [[ $COUNTERS -eq 0 ]]; then
  COUNTERS_ARGS+=(--pytorch_profiler)
fi

TORCHRUN_ARGS=(
  --nproc_per_node=1
  scripts/inference.py
  --architecture hf_pretrained
  --variant meta-llama/Llama-3.1-8B
  --tokenizer meta-llama/Llama-3.1-8B
  --num_batches 128
  --token "123"
  --num_tokens 256
)

echo "FMS_START $(date +%s)"
if [[ $COUNTERS -eq 1 ]]; then
  ${COUNTERS_ARGS[@]} torchrun ${TORCHRUN_ARGS[@]}
else
  torchrun ${TORCHRUN_ARGS[@]}
fi
echo "FMS_STOP $(date +%s)"

