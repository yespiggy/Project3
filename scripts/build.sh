#!/bin/bash
# Build all binaries into $P3_ROOT/bin. Run on the Vanda login node: bash scripts/build.sh
# The MPI+CUDA program is compiled with nvcc (MPI include dirs from mpicxx) and linked with mpicxx.
set -e
source "$(dirname "$0")/env.sh"
cd "$P3_ROOT"
mkdir -p bin build
ARCH=$(uname -m)
if [ "$ARCH" = x86_64 ]; then FLAGS="-O3 -march=x86-64-v3 -std=c++17"; else FLAGS="-O3 -mcpu=native -std=c++17"; fi
g++ $FLAGS -fopenmp tools/preprocess.cpp -o bin/preprocess
g++ $FLAGS -fopenmp tools/rmat_gen.cpp -o bin/rmat_gen
g++ $FLAGS src/serial/pagerank_serial.cpp -o bin/pagerank_serial
mpicxx $FLAGS -fopenmp src/cpu/pagerank_mpi_omp.cpp -o bin/pagerank_mpi_omp
echo "$ARCH" > bin/.arch
module load cuda12.4/toolkit/12.4.1 >/dev/null 2>&1 || true
if command -v nvcc >/dev/null; then
  nvcc -O3 -std=c++17 -arch=sm_86 src/gpu/pagerank_cuda.cu -lcusparse -o bin/pagerank_cuda
  MPI_INC=$(for d in $(mpicxx --showme:incdirs); do printf -- "-I%s " "$d"; done)
  CUDA_LIB=$(dirname "$(dirname "$(command -v nvcc)")")/lib64
  nvcc -O3 -std=c++17 -arch=sm_86 -Xcompiler -fopenmp $MPI_INC -c src/gpu/pagerank_mpi_cuda.cu -o build/pagerank_mpi_cuda.o
  mpicxx -fopenmp build/pagerank_mpi_cuda.o -L"$CUDA_LIB" -Wl,-rpath,"$CUDA_LIB" -lcudart -o bin/pagerank_mpi_cuda
else
  echo "nvcc not found: GPU binaries not built" >&2
fi
echo "built: $(ls bin | tr '\n' ' ')"
