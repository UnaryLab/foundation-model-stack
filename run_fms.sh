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

COUNTER_PROFILER=0
PYTORCH_PROFILER=1

COUNTERS_ARGS=(
        ncu
        -f
        --target-processes all
        --profile-from-start no
        -o memory
        --metrics
        sm__ops_path_tensor_src_bf16_dst_fp32.sum 
        dram__bytes_read.sum
        dram__bytes_write.sum
        --replay-mode application
)

MODEL=meta-llama/Llama-3.1-8B

TORCHRUN_ARGS=(
  --nproc_per_node=1
  scripts/inference.py
  --architecture hf_pretrained
  --variant ${MODEL}
  --tokenizer ${MODEL}
  --num_batches 128
  --token "123"
  --num_tokens 256
  --max_new_tokens 32
  --with_stack
)

if [[ $PYTORCH_PROFILER -eq 1 ]]; then
  TORCHRUN_ARGS+=(--pytorch_profiler)
fi

if [[ $COUNTER_PROFILER -eq 1 ]]; then
  TORCHRUN_ARGS+=(--ncu_profiler)
fi


echo "FMS_START $(date +%s)"
if [[ $COUNTER_PROFILER -eq 1 ]]; then
  ${COUNTERS_ARGS[@]} torchrun ${TORCHRUN_ARGS[@]}
else
  torchrun ${TORCHRUN_ARGS[@]}
fi
echo "FMS_STOP $(date +%s)"

