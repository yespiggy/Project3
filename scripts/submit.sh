#!/bin/bash
# Submit the experiment jobs from the project root. Usage:
#   bash scripts/submit.sh prep             # data preparation only (run once, after scripts/download_data.sh)
#   bash scripts/submit.sh cpu [after=ID]   # CPU jobs: 1 node (LJ, Friendster) + 2/4/6/8/10 nodes
#   bash scripts/submit.sh gpu [after=ID]   # GPU jobs: 2 GPUs on 1 node, 1 GPU on each of 2 nodes
#   bash scripts/submit.sh all              # prep, then everything else once prep has succeeded
# after=ID makes the jobs wait for job ID (e.g. the prep job) to finish successfully.
# Teammates sharing one data directory: export P3_DATA=/scratch/<owner>/p3data before submitting.
set -e -o pipefail
cd "$(dirname "$0")/../jobs"
what=${1:-cpu}
case "$what" in prep|cpu|gpu|all) ;; *) echo "usage: $0 prep|cpu|gpu|all [after=ID]" >&2; exit 2 ;; esac
# Conservative default: 144 cores fits the reported 160-core per-job limit.
CPU_NODES=${CPU_NODES:-2}
for nn in $CPU_NODES; do
  case "$nn" in 2|4|6|8|10) ;; *) echo "unsupported CPU_NODES: $nn" >&2; exit 2 ;; esac
done
dep=""
[ -n "${2:-}" ] && dep="-W depend=afterok:${2#after=}"
# Pass P3_DATA / P3_ROOT through to the jobs when they are set in this shell.
vars="P3_DATA=${P3_DATA:-/scratch/$USER/p3data},P3_ROOT=${P3_ROOT:-$HOME/project3},P3_RESULTS=${P3_RESULTS:-${P3_ROOT:-$HOME/project3}/results}"

sub() {
  local id
  id=$(qsub $dep ${P3_QUEUE:+-q "$P3_QUEUE"} -v "$vars${EXTRA:+,$EXTRA}" "$@") || return $?
  echo "$id"
}
submit_named() {
  local label=$1 id
  shift
  id=$(sub "$@") || return $?
  echo "$label: $id"
}

if [ "$what" = prep ] || [ "$what" = all ]; then
  id=$(sub prep.pbs)
  echo "prep: $id"
  [ "$what" = all ] && dep="-W depend=afterok:$id"
fi
if [ "$what" = cpu ] || [ "$what" = all ]; then
  submit_named "cpu 1 node LJ" cpu_n1_lj.pbs
  submit_named "cpu 1 node Friendster" cpu_n1_big.pbs
  for nn in $CPU_NODES; do
    EXTRA=NN=$nn submit_named "cpu $nn nodes" -N p3n$nn -l select=$nn:ncpus=72:mpiprocs=72:mem=120gb cpu_nodes.pbs
  done
fi
if [ "$what" = gpu ] || [ "$what" = all ]; then
  submit_named "gpu 1 node x 2 GPUs" gpu_1node.pbs
  submit_named "gpu 2 nodes x 1 GPU" gpu_2node.pbs
fi
qstat -u "$USER" 2> /dev/null || true

