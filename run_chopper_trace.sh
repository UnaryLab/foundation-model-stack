#!/bin/bash

set -eu

source fms_env

# make sure this matches run_fms.sh
PROFILER_TOK=30

CHOPPER_TRACE_ARGS=(
   -t kineto_trace.json
   -c ai.csv
   -nv
   --token $PROFILER_TOK
   -o ai.pkl
)

./chopper_trace.py ${CHOPPER_TRACE_ARGS[@]}
