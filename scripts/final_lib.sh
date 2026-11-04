# Helpers for the final experiment suite (jobs/final/*.pbs). Source after env.sh and run_lib.sh.
# Adds pure-MPI weak scaling, fixed-core rank x thread sweeps, --split breakdown runs and run bookkeeping
# on top of run_lib.sh. Every job writes into $P3_RESULTS (a fresh directory, see README) and records metadata.

# meta <tag>: record job, node, compiler and MPI information for the report's hardware/software table.
meta() {
  local f=$P3_RESULTS/meta_${1}_${PBS_JOBID:-local}.txt
  {
    echo "job=${PBS_JOBID:-local} name=${PBS_JOBNAME:-} date=$(date -Iseconds)"
    echo "nodes=${NODES[*]}"
    echo "git=$(git -C "$P3_ROOT" rev-parse --short HEAD 2> /dev/null || echo unknown)"
    lscpu | grep -E "Model name|^CPU\(s\)|Socket|Core\(s\) per|NUMA node\(s\)|L3"
    free -g | head -2
    g++ --version | head -1
    mpirun --version 2>&1 | head -1
    module list 2>&1 | tr '\n' ' '
    echo
    command -v nvidia-smi > /dev/null && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo "infiniband devices: $(ls /sys/class/infiniband 2> /dev/null | tr '\n' ' ')"
    ompi_info 2> /dev/null | grep -E "MCA (btl|pml|mtl):" | awk '{print $2, $3}' | tr '\n' ' '
    echo
  } > "$f" 2>&1
}

# weak_mode <mode> <csv> <cores...>: E2 weak scaling on rmat_p<cores>.bin in the given mode (hybrid | mpi).
weak_mode() {
  local mode=$1 csv=$2
  shift 2
  for c in "$@"; do
    [ "$c" = 1 ] && [ "$mode" = mpi ] && continue  # identical to the hybrid 1-core run
    read n rpn th <<< "$(layout $c $mode)"
    run_mpi $n $rpn $th $csv $BIN/pagerank_mpi_omp --graph $P3_DATA/rmat_p$c.bin --part edge --comm ghost \
      $PERF --exp E2_$mode
    echo "E2 $mode $c cores done $(date)"
  done
}

# sweep <graph.bin> <csv> <nodes> <ranks_per_node...>: fixed core count (nodes x 72), varying ranks per node
# (threads per rank = 72 / ranks per node). Tests the MPI process / OpenMP thread trade-off (E7).
sweep() {
  local g=$1 csv=$2 n=$3
  shift 3
  for rpn in "$@"; do
    run_mpi $n $rpn $((CPN / rpn)) $csv $BIN/pagerank_mpi_omp --graph $g --part edge --comm ghost $PERF --exp E7
    echo "E7 $(basename $g .bin) $n nodes x $rpn ranks done $(date)"
  done
}

# split_runs <graph.bin> <csv> <cores>: hybrid and pure-MPI runs with --split 1 (barrier before each communication
# call; t_wait = synchronisation / imbalance wait, t_comm = data movement only). Diagnostic only (E8).
split_runs() {
  local g=$1 csv=$2 c=$3
  for mode in hybrid mpi; do
    [ "$c" = 1 ] && [ "$mode" = mpi ] && continue
    read n rpn th <<< "$(layout $c $mode)"
    run_mpi $n $rpn $th $csv $BIN/pagerank_mpi_omp --graph $g --part edge --comm ghost $PERF --split 1 --exp E8_$mode
  done
  echo "E8 $(basename $g .bin) $c cores done $(date)"
}

# transport_runs <graph.bin> <csv> <cores>: E9, pure MPI at <cores> ranks with the InfiniBand verbs transport
# explicitly enabled (OpenMPI 4.1 disables btl/openib on InfiniBand by default when UCX is absent, which leaves
# inter-node traffic on btl/tcp). Compared against the default-transport rows of the same configuration.
IB_MCA="--mca btl self,vader,openib --mca btl_openib_allow_ib 1 --mca btl_openib_warn_no_device_params_found 0"
transport_runs() {
  local g=$1 csv=$2 c=$3 save=$MPIRUN
  read n rpn th <<< "$(layout $c mpi)"
  MPIRUN="$save $IB_MCA"
  run_mpi $n $rpn $th $csv $BIN/pagerank_mpi_omp --graph $g --part edge --comm ghost $PERF --exp E9_openib
  read n rpn th <<< "$(layout $c hybrid)"
  run_mpi $n $rpn $th $csv $BIN/pagerank_mpi_omp --graph $g --part edge --comm ghost $PERF --exp E9_openib_hybrid
  MPIRUN=$save
  echo "E9 $(basename $g .bin) $c cores done $(date)"
}

# run_mpi is wrapped so that a failing configuration (e.g. 720 pure-MPI ranks) is logged with its layout and
# the job continues with the remaining configurations. Failures are listed in $P3_RESULTS/failures.txt and must be
# reported in the write-up, not silently dropped; a failed run writes no CSV row.
eval "$(declare -f run_mpi | sed '1s/^run_mpi/_run_mpi_orig/')"
run_mpi() {
  local st=0
  _run_mpi_orig "$@" || st=$?
  if [ "$st" != 0 ]; then
    echo "FAILED (exit $st): nodes=$1 ranks/node=$2 threads=$3 csv=$(basename "$4") cmd=${*:5} job=${PBS_JOBID:-local} $(date)" \
      | tee -a "$P3_RESULTS/failures.txt" >&2
  fi
  return 0
}
