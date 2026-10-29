# Shared environment for building and running on NUS HPC Vanda. Source this file.
export P3_ROOT=${P3_ROOT:-$HOME/project3}
export P3_DATA=${P3_DATA:-/scratch/$USER/p3data}
export P3_RESULTS=${P3_RESULTS:-$P3_ROOT/results}
module purge >/dev/null 2>&1
module load gcc/13.1.0 openmpi/gcc/64/4.1.8 >/dev/null 2>&1
