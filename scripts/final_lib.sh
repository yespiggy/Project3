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

# reset_outputs <files...>: run_mpi and the programs APPEND to their CSV files (the header is written only when the file
# is missing). Re-running a job, or running two jobs into the same directory, would therefore mix two runs in one
# file with duplicated rep numbers. Every job calls this first with the files it writes: existing ones are moved to
# $P3_RESULTS/superseded/ (kept, never deleted) so each CSV holds exactly one run.
reset_outputs() {
  local f
  for f in "$@"; do
    [ -e "$f" ] || continue
    mkdir -p "$P3_RESULTS/superseded"
    mv "$f" "$P3_RESULTS/superseded/$(basename "$f").$(date +%Y%m%d-%H%M%S)-${PBS_JOBID:-local}"
    echo "moved existing $f to $P3_RESULTS/superseded/"
  done
}

# py_ready: true when the system python3 can import what the validation / smoke scripts need.
py_ready() { python3 -c "import numpy, scipy, networkx" > /dev/null 2>&1; }

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

# Guard for large pure-MPI runs over btl/tcp (needed by f4_rerun.pbs).
# In the first 8- and 10-node f3 jobs, a 576/720-rank run that started less than 60 s after the previous large run
# failed with "btl_tcp_endpoint_start_connect: bind on local address ... failed: Address already in use (98)".
# Every rank opens a TCP connection to almost every off-node rank, and the sockets of the previous run were still in
# TIME_WAIT (60 s on Linux), so the node had no free local ports. The same configurations succeeded when started a
# few minutes later. Therefore: before every run with more than 144 ranks, wait until the TIME_WAIT sockets on all
# nodes of the run have drained; after a failed run, kill leftover processes on those nodes.
tw_max() {  # tw_max <nodes>: largest number of TIME_WAIT TCP sockets on the first <nodes> nodes of the job
  local n=$1 i c max=0
  for ((i = 0; i < n; i++)); do
    c=$(ssh -o BatchMode=yes "${NODES[$i]}" "cat /proc/net/tcp /proc/net/tcp6 2> /dev/null | awk '\$4 == \"06\"' | wc -l" \
      2> /dev/null || echo 0)
    ((c > max)) && max=$c
  done
  echo "$max"
}
wait_ports() {  # wait_ports <nodes>: wait (at most 5 min) until every node has fewer than 500 TIME_WAIT sockets
  local n=$1 t0=$SECONDS m
  while :; do
    m=$(tw_max "$n")
    ((m < 500 || SECONDS - t0 > 300)) && break
    sleep 15
  done
  echo "ports: max TIME_WAIT $m on $n nodes after $((SECONDS - t0)) s wait $(date)"
}
clean_nodes() {  # clean_nodes <nodes>: kill processes left behind by an aborted mpirun
  local n=$1 i
  for ((i = 0; i < n; i++)); do
    ssh -o BatchMode=yes "${NODES[$i]}" "pkill -u \$USER -f '$BIN/pagerank_mpi_omp'; pkill -u \$USER -x orted" \
      2> /dev/null || true
  done
  sleep 20
}
eval "$(declare -f run_mpi | sed '1s/^run_mpi/_run_mpi_logged/')"
run_mpi() {
  local nf
  (($1 * $2 > 144)) && wait_ports "$1"
  nf=$(wc -l < "$P3_RESULTS/failures.txt" 2> /dev/null || echo 0)
  _run_mpi_logged "$@"
  if [ "$(wc -l < "$P3_RESULTS/failures.txt" 2> /dev/null || echo 0)" -gt "$nf" ]; then
    clean_nodes "$1"
  fi
  return 0
}
