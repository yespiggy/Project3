#!/bin/bash
# Build all binaries into $P3_ROOT/bin. Run on the Vanda login node: bash scripts/build.sh
set -e
source "$(dirname "$0")/env.sh"
cd "$P3_ROOT"
mkdir -p bin
FLAGS="-O3 -march=x86-64-v3 -std=c++17"
g++ $FLAGS -fopenmp tools/preprocess.cpp -o bin/preprocess
g++ $FLAGS -fopenmp tools/rmat_gen.cpp -o bin/rmat_gen
g++ $FLAGS src/serial/pagerank_serial.cpp -o bin/pagerank_serial
mpicxx $FLAGS -fopenmp src/cpu/pagerank_mpi_omp.cpp -o bin/pagerank_mpi_omp
module load cuda12.4/toolkit/12.4.1 >/dev/null 2>&1
nvcc -O3 -std=c++17 -arch=sm_86 src/gpu/pagerank_cuda.cu -lcusparse -o bin/pagerank_cuda
echo "built: $(ls bin | tr '\n' ' ')"
