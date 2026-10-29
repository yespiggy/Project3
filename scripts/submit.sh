#!/bin/bash
# Submit the experiment jobs from the project root. Usage:
#   bash scripts/submit.sh prep             # data preparation only (run once, after scripts/download_data.sh)
#   bash scripts/submit.sh cpu [after=ID]   # CPU jobs: 1 node (LJ, Friendster) + 2/4/6/8/10 nodes
#   bash scripts/submit.sh gpu [after=ID]   # GPU jobs: 2 GPUs on 1 node, 1 GPU on each of 2 nodes
#   bash scripts/submit.sh all              # prep, then everything else once prep has succeeded
# after=ID makes the jobs wait for job ID (e.g. the prep job) to finish successfully.
# Teammates sharing one data directory: export P3_DATA=/scratch/<owner>/p3data before submitting.
set -e
cd "$(dirname "$0")/../jobs"
what=${1:-all}
dep=""
[ -n "$2" ] && dep="-W depend=afterok:${2#after=}"
# Pass P3_DATA / P3_ROOT through to the jobs when they are set in this shell.
vars="P3_DATA=${P3_DATA:-/scratch/$USER/p3data},P3_ROOT=${P3_ROOT:-$HOME/project3}"

sub() { qsub $dep -v "$vars${EXTRA:+,$EXTRA}" "$@"; }

if [ "$what" = prep ] || [ "$what" = all ]; then
  id=$(qsub -v "$vars" prep.pbs)
  echo "prep: $id"
  [ "$what" = all ] && dep="-W depend=afterok:$id"
fi
if [ "$what" = cpu ] || [ "$what" = all ]; then
  echo "cpu 1 node LJ:         $(sub cpu_n1_lj.pbs)"
  echo "cpu 1 node Friendster: $(sub cpu_n1_big.pbs)"
  for nn in 2 4 6 8 10; do
    echo "cpu $nn nodes:          $(EXTRA=NN=$nn sub -N p3n$nn -l select=$nn:ncpus=72:mpiprocs=72:mem=120gb cpu_nodes.pbs)"
  done
fi
if [ "$what" = gpu ] || [ "$what" = all ]; then
  echo "gpu 1 node x 2 GPUs:   $(sub gpu_1node.pbs)"
  echo "gpu 2 nodes x 1 GPU:   $(sub gpu_2node.pbs)"
fi
qstat -u "$USER" 2> /dev/null || true
