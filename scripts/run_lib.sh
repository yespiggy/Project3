# Helpers for launching runs inside a PBS job. Source after env.sh.
# This OpenMPI build is not PBS-aware: we pass an explicit hostfile and --prefix so remote orted is found.
# Each rank is bound to its own block of cores (PE=threads) and OpenMP pins threads inside that block.
# Without the binding, OMP_PROC_BIND puts every rank's master thread on core 0 and the ranks serialize.
NODES=($(sort -u "$PBS_NODEFILE"))
MPIRUN="mpirun --prefix $(dirname "$(dirname "$(which mpirun)")") -x LD_LIBRARY_PATH -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES --bind-to core --mca btl_vader_single_copy_mechanism none"
HOSTDIR=${TMPDIR:-/tmp}/p3hosts.$$
mkdir -p "$HOSTDIR"

# csv_flag <file>: print "--header 1" when the CSV does not exist yet.
csv_flag() { [ -s "$1" ] || echo "--header 1"; }

# run_mpi <nodes> <ranks_per_node> <threads> <csv> <program args...>
run_mpi() {
  local n=$1 rpn=$2 th=$3 csv=$4
  shift 4
  local hf=$HOSTDIR/h_${n}_${rpn}
  for ((i = 0; i < n; i++)); do echo "${NODES[$i]} slots=$rpn"; done > "$hf"
  OMP_NUM_THREADS=$th OMP_PROC_BIND=close OMP_PLACES=cores \
    $MPIRUN --hostfile "$hf" -np $((n * rpn)) --map-by ppr:$rpn:node:PE=$th "$@" $(csv_flag "$csv") >> "$csv"
}

# layout <cores> <mode>: echo "nodes ranks_per_node threads" for 16-core chunks.
# mode=mpi -> one rank per core; mode=hybrid -> one rank per node, threads fill the node.
layout() {
  local c=$1 mode=$2 n=$(( ($1 + 15) / 16 ))
  if [ "$mode" = mpi ]; then echo "$n $((c / n)) 1"; else echo "$n 1 $((c / n))"; fi
}
