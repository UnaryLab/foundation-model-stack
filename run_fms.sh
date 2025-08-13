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

set -eux

cd $SLURM_SUBMIT_DIR
source fms_env

run_fms() {
  local COUNTER_PROFILER=$1
  local PYTORCH_PROFILER=$2

  local OUTPUT_NAME=$3
  local COUNTERS_ARGS=(
          ncu
          -f
          --target-processes all
          --profile-from-start no
          -o ${OUTPUT_NAME}_ai
          --metrics sm__ops_path_tensor_src_bf16_dst_fp32.sum,dram__bytes_read.sum,dram__bytes_write.sum
          --replay-mode application
  )

  local MODEL=meta-llama/Llama-3.1-8B

  local MAX_TOK=$4
  local PROFILER_TOK=$5

  local TORCHRUN_ARGS=(
    --nproc_per_node=1
    scripts/inference.py
    --architecture hf_pretrained
    --variant ${MODEL}
    --tokenizer ${MODEL}
    --num_batches $6
    --token "123"
    --num_tokens $7
    --max_new_tokens ${MAX_TOK}
    # --with_stack
  )

  if [[ $PYTORCH_PROFILER -eq 1 ]]; then
    TORCHRUN_ARGS+=(
      --pytorch_profiler
      --pytorch_profiler_output ${OUTPUT_NAME}
    )
  fi

  if [[ $COUNTER_PROFILER -eq 1 ]]; then
    TORCHRUN_ARGS+=(
      --ncu_profiler
      --ncu_profiler_token ${PROFILER_TOK}
    )
  fi


  if [[ $COUNTER_PROFILER -eq 1 ]]; then
    ${COUNTERS_ARGS[@]} torchrun ${TORCHRUN_ARGS[@]}
    ncu --import ${OUTPUT_NAME}_ai.ncu-rep --csv --page raw > ${OUTPUT_NAME}.csv
  else
    torchrun ${TORCHRUN_ARGS[@]}
  fi
}

echo "FMS_START $(date +%s)"

BATCH_SIZE=128
PROMPT_LEN=256
GEN=32

COUNTERS=1
PYTORCH_PROFILER=0
# counters for prefill
PROF_TOK=0
run_fms ${COUNTERS} ${PYTORCH_PROFILER} "g${GEN}_tok${PROF_TOK}_b${BATCH_SIZE}_p${PROMPT_LEN}" ${GEN} ${PROF_TOK} ${BATCH_SIZE} ${PROMPT_LEN}
# counters for decode tok10
PROF_TOK=10
run_fms ${COUNTERS} ${PYTORCH_PROFILER} "g${GEN}_tok${PROF_TOK}_b${BATCH_SIZE}_p${PROMPT_LEN}" ${GEN} ${PROF_TOK} ${BATCH_SIZE} ${PROMPT_LEN}
# counters for decode tok30
PROF_TOK=30
run_fms ${COUNTERS} ${PYTORCH_PROFILER} "g${GEN}_tok${PROF_TOK}_b${BATCH_SIZE}_p${PROMPT_LEN}" ${GEN} ${PROF_TOK} ${BATCH_SIZE} ${PROMPT_LEN}
COUNTERS=0
PYTORCH_PROFILER=1
# pytorch trace for all tokens
run_fms ${COUNTERS} ${PYTORCH_PROFILER} "g${GEN}_b${BATCH_SIZE}_p${PROMPT_LEN}" ${GEN} ${PROF_TOK} ${BATCH_SIZE} ${PROMPT_LEN}
echo "FMS_STOP $(date +%s)"

