#!/bin/bash

set -eu

source fms_env

TIMESTAMPS=$(fd 'g\d+_b\d+_p\d+.json' . -x basename)
GEN=$(echo $TIMESTAMPS | rg 'g(\d+)_b\d+_p\d+.json' -r '$1')
BATCH_SIZE=$(echo $TIMESTAMPS | rg 'g\d+_b(\d+)_p\d+.json' -r '$1')
PROMPT_LEN=$(echo $TIMESTAMPS | rg 'g\d+_b\d+_p(\d+).json' -r '$1')
echo timestamp file: $TIMESTAMPS
echo generated: $GEN
echo batch size: $BATCH_SIZE
echo prompt len: $PROMPT_LEN

echo "Generating timestamp pickle..."
CHOPPER_TRACE_ARGS=(
   -t $TIMESTAMPS
   -nv
   -o g${GEN}_b${BATCH_SIZE}_p${PROMPT_LEN}.pkl
)
./chopper_trace.py ${CHOPPER_TRACE_ARGS[@]}

TOKS=$(fd "g${GEN}_tok\d+_b${BATCH_SIZE}_p${PROMPT_LEN}.csv" . -x basename | rg "g${GEN}_tok(\d+)_b${BATCH_SIZE}_p${PROMPT_LEN}.csv" -r '$1' | sort -n)
echo found toks: $TOKS

for TOK in $TOKS
do
   echo "Adding counters for token: $TOK..."
   CHOPPER_TRACE_ARGS=(
      -p g${GEN}_b${BATCH_SIZE}_p${PROMPT_LEN}.pkl
      -c g${GEN}_tok${TOK}_b${BATCH_SIZE}_p${PROMPT_LEN}.csv
      -nv
      --token $TOK
      -o g${GEN}_b${BATCH_SIZE}_p${PROMPT_LEN}.pkl
   )

   ./chopper_trace.py ${CHOPPER_TRACE_ARGS[@]}
done
echo "All done! generated: g${GEN}_b${BATCH_SIZE}_p${PROMPT_LEN}.pkl"
