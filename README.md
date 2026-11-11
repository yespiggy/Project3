# Distributed PageRank with MPI+OpenMP and MPI+CUDA (DSA5208 Project 3)

Ma Xiaoxiao (A0259173X), Ye Sifan (A0262394Y), Chen Yuda (A0353182H)

The report is [`report/report.pdf`](report/report.pdf). This repository contains everything needed to reproduce it:
the source code of the serial, MPI+OpenMP, CUDA and MPI+CUDA programs, the PBS job scripts that produced every
measurement, the raw result files, and the script that turns those files into the report's figures and tables.

## Layout

| Path | Contents |
|---|---|
| `src/serial/pagerank_serial.cpp` | serial reference implementation (pull-based PageRank on an in-edge CSR) |
| `src/cpu/pagerank_mpi_omp.cpp` | MPI+OpenMP implementation: vertex or edge partition, allgather or ghost exchange |
| `src/common/` | binary graph format, CSV output, 1D partition and ghost-exchange plan (`dist.hpp`) |
| `src/gpu/pagerank_cuda.cu` | single-GPU CUDA implementation: thread / warp / degree-bucket kernels, cuSPARSE reference |
| `src/gpu/pagerank_mpi_cuda.cu` | MPI+CUDA multi-GPU implementation (one rank per GPU, 64-bit offsets) |
| `tools/preprocess.cpp` | SNAP edge list to binary in-edge CSR (renumbering, self-loop and duplicate removal) |
| `tools/rmat_gen.cpp` | R-MAT synthetic graph generator (Graph500 parameters) |
| `tools/stream_bw.cpp`, `tools/gpu_bw.cu` | CPU (STREAM) and GPU (copy, triad, random gather) memory-bandwidth probes |
| `tools/smoke.py`, `tools/validate.py`, `tools/compare_vec.py` | correctness tests (NetworkX, serial reference, at scale) |
| `tools/analyze_google.py` | data findings for web-Google |
| `tools/make_report.py` | builds `report/figures/*.pdf` and `report/generated/*.tex` from `results_final/` |
| `tools/aggregate.py` | alternative summary of a results directory as JSON |
| `jobs/final/*.pbs` | the PBS jobs that produced `results_final/` (see below) |
| `scripts/` | environment, build, data download and job submission |
| `results_final/` | raw CSV files of every run, hardware metadata of every job, validation results |
| `report/` | LaTeX source of the report, generated figures and tables |

No BLAS, LAPACK or sparse library is used by our implementations. cuSPARSE appears only as a reference kernel in
`pagerank_cuda.cu`.

## Building (NUS Vanda)

```bash
export P3_ROOT=$PWD                # the repository root on Vanda
bash scripts/build.sh              # GCC 13.1, OpenMPI 4.1.8, CUDA 12.4; binaries go to bin/
```

## Running the programs

Every program reads a graph in the binary in-edge CSR format written by `bin/preprocess` or `bin/rmat_gen` (see Data)
and prints one CSV row per timed repetition; `--header 1` prints the column names first.

```bash
G=/scratch/$USER/p3data/soc-LiveJournal1.bin

# serial program, run to convergence
bin/pagerank_serial --graph $G --iters 200 --tol 1e-10 --header 1

# MPI only (pure MPI): 72 single-threaded ranks on one node
OMP_NUM_THREADS=1 mpirun -np 72 --bind-to core \
  bin/pagerank_mpi_omp --graph $G --part edge --comm ghost --iters 50 --tol 0 --reps 5 --header 1

# MPI+OpenMP (hybrid layout): one rank per NUMA domain, 36 threads per rank
OMP_NUM_THREADS=36 OMP_PROC_BIND=close OMP_PLACES=cores mpirun -np 2 --map-by ppr:2:node:PE=36 --bind-to core \
  bin/pagerank_mpi_omp --graph $G --part edge --comm ghost --iters 50 --tol 0 --reps 5 --header 1

# CUDA on one GPU: all four SpMV kernels (thread, warp, hybrid = degree buckets, cusparse)
bin/pagerank_cuda --graph $G --kernel all --iters 50 --tol 0 --reps 5 --header 1

# MPI+CUDA: one rank per GPU, here two GPUs on one node
mpirun -np 2 bin/pagerank_mpi_cuda --graph $G --kernel thread,hybrid --iters 50 --tol 0 --reps 5 --header 1
```

Options of `pagerank_mpi_omp`: `--part vertex|edge` (equal vertices or equal in-edges per rank), `--comm
allgather|ghost` (exchange method), `--split 1` (barrier before every MPI call, reports the wait separately),
`--out r.bin` (write the PageRank vector), `--log resid.csv` (residual of every iteration). On several nodes the job
scripts add a host file and the rank placement (`scripts/run_lib.sh`, function `run_mpi`).

## Data

```bash
bash scripts/download_data.sh      # login node: web-Google, soc-LiveJournal1, com-friendster into /scratch/$USER/p3data
```

The preprocessing job (`p0_prep.pbs`) converts the SNAP files to binary CSR and generates the R-MAT graphs:
`rmat_p<k>` (131,072·k vertices, edge factor 16, seed k) for weak scaling and `rmat_n20_ef<f>` (2^20 vertices, edge
factor f) for the serial size study. Graphs are reproducible from the seed and the generator's thread count (32).

## Running the experiments

```bash
bash scripts/final_submit.sh all   # prep, then all experiment jobs, then post (validation + aggregation)
```

No queue is specified; Vanda routes the jobs. Every CPU job requests whole 72-core nodes.

| Job | Resources | Experiments |
|---|---|---|
| `p0_prep.pbs` | 1 node | build, preprocessing, R-MAT generation, small-graph tests against NetworkX |
| `f1_serial.pbs` | 1 node | validation references, STREAM, serial size study (S_ef, S_rmat, S_real) |
| `f2_node1.pbs` | 1 node | 1–72 cores: strong scaling (E1), weak scaling (E2), ranks×threads sweep (E7), `--split` (E8) |
| `f3_nodes.pbs` | 2, 4, 6, 8, 10 nodes | 144–720 cores: E1, E2, E8, validation at scale, InfiniBand-verbs test (E9); 4 nodes: partition × exchange (E3), E7 |
| `f4_rerun.pbs` | 8, 10 nodes | rerun of the 576/720-rank configurations that failed in `f3` (nodes out of TCP ports; see `scripts/final_lib.sh`) |
| `f5_ibprobe.pbs` | 2 nodes | probe of why the InfiniBand-verbs transport (E9) cannot start on Vanda (Section 4 of the report, Platform) |
| `g1_gpu1node.pbs` | 1 GPU node | GPU bandwidth probe, GPU validation, single-GPU kernels (E4), MPI+CUDA on 1–2 GPUs (E6) |
| `g2_gpu2node.pbs` | 2 GPU nodes | MPI+CUDA with one GPU on each of two nodes (E6) |
| `post.pbs` | 1 node | validation and aggregation after all jobs |

Every program appends one CSV row per timed repetition (columns defined in `src/common/graph.hpp`). Every configuration
used in the report has its full set of repetitions. The 576/720-rank configurations that failed in `f3` were rerun
successfully by `f4`; their failed attempts wrote no rows. The E9 runs (MPI over InfiniBand verbs) cannot start on
Vanda and produced no rows, so their empty CSV files are not included; MPI traffic between nodes uses TCP.

## Rebuilding the report

```bash
python3 tools/make_report.py results_final report   # figures and tables (needs matplotlib)
cd report && tectonic report.tex                    # or: latexmk -pdf report.tex
```

`make_report.py` reads only `results_final/`. It writes the eight figures of the report to `report/figures/`, the
table bodies to `report/generated/tab_*.tex`, and every measured number quoted in the text to
`report/generated/numbers.tex`, so the text, the tables and the figures always agree with the raw files.

## Branches

`submission` is the version to be graded. `main`, `feedback-72core-multigpu`, `final-experiments` and
`vanda-timing-validation-720core` are earlier states of the project and are kept for their history. Their
measurements were taken with earlier versions of the code; the same experiments were repeated with the final code
and are the ones in `results_final/`.
