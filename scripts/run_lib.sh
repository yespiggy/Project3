# Helpers for launching runs inside a PBS job. Source after env.sh.
# This OpenMPI build is not PBS-aware: we pass an explicit hostfile and --prefix so remote orted is found.
# Each rank is bound to its own block of cores (PE=threads) and OpenMP pins threads inside that block.
# Without the binding, OMP_PROC_BIND puts every rank's master thread on core 0 and the ranks serialize.
NODES=($(sort -u "$PBS_NODEFILE"))
HOSTDIR=$P3_ROOT/tmp/${PBS_JOBID:-$$}  # on the shared filesystem: other nodes write into it (GPU map)
mkdir -p "$HOSTDIR"
export P3_GPUMAP=$HOSTDIR/gpus
MPIRUN="mpirun --prefix $(dirname "$(dirname "$(which mpirun)")") -x LD_LIBRARY_PATH -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES -x P3_GPUMAP --bind-to core --mca btl_vader_single_copy_mechanism none"
BIN=$P3_ROOT/bin

# Cores per CPU node (cn nodes: 72) and cores per NUMA domain, used by layout().
CPN=${CPN:-72}
NUMA=$(lscpu | awk -F: '/^NUMA node\(s\)/ {gsub(/ /, "", $2); print $2}')
NUMA=${NUMA:-1}
CPNUMA=$((CPN / NUMA))

# Core counts for strong and weak scaling: within one node, then whole nodes (72 cores each) up to 10 nodes.
CORES_1NODE="1 2 4 8 16 36 72"

# csv_flag <file>: print "--header 1" when the CSV does not exist yet.
csv_flag() { [ -s "$1" ] || echo "--header 1"; }

# check_node: print the node's CPU topology and stop the job if it is not a 72-core cn node.
check_node() {
  echo "nodes: ${NODES[*]}"
  lscpu | grep -E "^Architecture|Model name|^CPU\(s\)|Socket|Core\(s\) per|Thread\(s\) per|NUMA node\(s\)"
  free -g | head -2
  local ncpu
  ncpu=$(lscpu | awk -F: '/^CPU\(s\)/ {gsub(/ /, "", $2); print $2}')
  if [ "${ncpu:-0}" -lt "$CPN" ]; then
    echo "ERROR: node has $ncpu CPUs, expected a $CPN-core cn node; check the select statement" >&2
    exit 1
  fi
  echo "layout: $CPN cores/node, $NUMA NUMA domains, $CPNUMA cores per domain"
}

# run_mpi <nodes> <ranks_per_node> <threads> <csv> <program args...>
run_mpi() {
  local n=$1 rpn=$2 th=$3 csv=$4
  shift 4
  local hf=$HOSTDIR/h_${n}_${rpn}
  for ((i = 0; i < n; i++)); do echo "${NODES[$i]} slots=$rpn"; done > "$hf"
  OMP_NUM_THREADS=$th OMP_PROC_BIND=close OMP_PLACES=cores \
    $MPIRUN --hostfile "$hf" -np $((n * rpn)) --map-by ppr:$rpn:node:PE=$th "$@" $(csv_flag "$csv") >> "$csv"
}

# layout <cores> <mode>: echo "nodes ranks_per_node threads".
# Cores fill whole 72-core nodes first (c <= 72 runs on one node).
# mode=mpi    -> one rank per core.
# mode=hybrid -> one rank per NUMA domain when the node is filled by whole domains, otherwise one rank per node;
#                OpenMP threads fill the rest.
layout() {
  local c=$1 mode=$2 n=$(( ($1 + CPN - 1) / CPN ))
  local k=$((c / n))
  if [ "$mode" = mpi ]; then
    echo "$n $k 1"
  else
    local rpn=1
    if [ "$k" -ge "$CPNUMA" ] && [ $((k % CPNUMA)) -eq 0 ]; then rpn=$((k / CPNUMA)); fi
    echo "$n $rpn $((k / rpn))"
  fi
}

# strong <graph.bin> <csv> <cores...>: E1 strong scaling, pure MPI and hybrid at each core count.
# At 1 core the two modes are the same run, so only the hybrid one is made. Uses $PERF.
strong() {
  local g=$1 csv=$2
  shift 2
  for c in "$@"; do
    for mode in hybrid mpi; do
      [ "$c" = 1 ] && [ "$mode" = mpi ] && continue
      read n rpn th <<< "$(layout $c $mode)"
      run_mpi $n $rpn $th $csv $BIN/pagerank_mpi_omp --graph $g --part edge --comm ghost $PERF --exp E1_$mode
    done
    echo "E1 $(basename $g .bin) $c cores done $(date)"
  done
}

# weak <csv> <cores...>: E2 weak scaling on rmat_p<cores>.bin (about 2M edges per core), hybrid only.
weak() {
  local csv=$1
  shift
  for c in "$@"; do
    read n rpn th <<< "$(layout $c hybrid)"
    run_mpi $n $rpn $th $csv $BIN/pagerank_mpi_omp --graph $P3_DATA/rmat_p$c.bin --part edge --comm ghost \
      $PERF --exp E2
    echo "E2 $c cores done $(date)"
  done
}

# gpu_map: record which GPU(s) PBS assigned on every node of the job. Ranks started over ssh on other nodes
# do not inherit CUDA_VISIBLE_DEVICES, so scripts/gpu_wrap.sh reads it back from this map.
gpu_map() {
  mkdir -p "$P3_GPUMAP"
  if command -v pbsdsh > /dev/null; then
    pbsdsh -- bash -c "echo \${CUDA_VISIBLE_DEVICES:-} > $P3_GPUMAP/\$(hostname -s)" || true
  fi
  [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && echo "$CUDA_VISIBLE_DEVICES" > "$P3_GPUMAP/$(hostname -s)"
  for h in "${NODES[@]}"; do
    local s=${h%%.*}
    echo "GPU map $s: $(cat "$P3_GPUMAP/$s" 2> /dev/null || echo '(not set)')"
    ssh -o BatchMode=yes "$h" nvidia-smi --query-gpu=index,pci.bus_id,name,utilization.gpu,memory.used \
      --format=csv,noheader 2> /dev/null | sed "s/^/  $s: /"
  done
}

# ensure_build: rebuild if the binaries were compiled for a different CPU architecture than this node.
ensure_build() {
  if [ "$(cat "$BIN/.arch" 2> /dev/null)" != "$(uname -m)" ]; then
    echo "binaries missing or built for another architecture; rebuilding on $(hostname -s)"
    bash "$P3_ROOT/scripts/build.sh"
  fi
}
