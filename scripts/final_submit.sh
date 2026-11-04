#!/bin/bash
# Submit the final experiment suite (jobs/final). Run on the Vanda login node from the project root:
#   bash scripts/final_submit.sh all                  # prep, then f1, f2, f3 (2 4 6 8 10 nodes), g1, g2 after prep,
#                                                     # then post (validation + aggregation) after all of them ended
#   bash scripts/final_submit.sh nodes "4 10"         # only the 4- and 10-node CPU jobs (data must exist)
#   AFTER=<jobid> bash scripts/final_submit.sh gpu    # make the jobs wait for job <jobid> to succeed
#   GPU_QUEUE=<queue> bash scripts/final_submit.sh gpu   # send the two GPU jobs to an explicit queue
#   bash scripts/final_submit.sh post                 # only the validation / aggregation job (no dependency)
# P3_DATA (graphs) defaults to /scratch/$USER/p3data; the SNAP text files must already be there
# (scripts/download_data.sh). No queue is given for CPU jobs: Vanda's default routing picks it (advice of Prof. Chen).
# GPU jobs use the queue named in GPU_QUEUE if it is set, otherwise the default routing as well.
# Results go to $P3_RESULTS (default results_final/); each job writes its own CSV files (suffix _n<nodes>) and moves
# any older copy of those files to $P3_RESULTS/superseded/ first, so re-submitting never mixes two runs in one CSV.
set -e -o pipefail
cd "$(dirname "$0")/.."
export P3_ROOT=$PWD
export P3_DATA=${P3_DATA:-/scratch/$USER/p3data}
export P3_RESULTS=${P3_RESULTS:-$P3_ROOT/results_final}
mkdir -p "$P3_RESULTS"
what=${1:-all}
NNS=${2:-"2 4 6 8 10"}
vars="P3_ROOT=$P3_ROOT,P3_DATA=$P3_DATA,P3_RESULTS=$P3_RESULTS"
cd "$P3_ROOT/jobs/final"
dep=${AFTER:+-W depend=afterok:$AFTER}
JOBIDS=()
sub() { qsub $dep -v "$vars${EXTRA:+,$EXTRA}" "$@"; }
log() {
  echo "$(date -Iseconds) $1 $2" | tee -a "$P3_RESULTS/submitted.txt"
  JOBIDS+=("$2")
}
if [ "$what" = all ] || [ "$what" = prep ]; then
  id=$(sub p0_prep.pbs)
  log prep "$id"
  dep="-W depend=afterok:$id"
fi
case "$what" in serial|all) log serial "$(sub f1_serial.pbs)" ;; esac
case "$what" in node1|all) log node1 "$(sub f2_node1.pbs)" ;; esac
case "$what" in nodes|all)
  for nn in $NNS; do
    wt=01:00:00
    [ "$nn" = 4 ] && wt=01:45:00  # the 4-node job also runs the E3 design comparison and the E7 sweep
    log "nodes$nn" "$(EXTRA=NN=$nn sub -N p3f_n$nn -l select=$nn:ncpus=72:mpiprocs=72:mem=200gb -l walltime=$wt f3_nodes.pbs)"
  done ;;
esac
case "$what" in gpu|all)
  log gpu1 "$(sub ${GPU_QUEUE:+-q $GPU_QUEUE} g1_gpu1node.pbs)"
  log gpu2 "$(sub ${GPU_QUEUE:+-q $GPU_QUEUE} g2_gpu2node.pbs)" ;;
esac
# post runs after every job submitted here has ended, whatever their exit status (afterany)
case "$what" in post|all)
  after=""
  for j in "${JOBIDS[@]}"; do after="$after:$j"; done
  dep=${after:+-W depend=afterany$after}
  log post "$(sub post.pbs)" ;;
esac
qstat -u "$USER" 2> /dev/null || true
