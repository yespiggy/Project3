# Parallel PageRank: feedback revision

Based on `yespiggy/Project3`, branch `feedback-72core-multigpu`, commit
`20c289ad6745946d0a363490eca9a49300f22dfc`. This folder contains the source,
job scripts and analysis tools; historical benchmark outputs are deliberately
not included. Use a fresh results directory for the changed timing definitions.

## Vanda setup and queue inspection

Place this folder in a new directory on Vanda, then set:

```bash
export P3_ROOT="$PWD"
export P3_DATA=/scratch/$USER/p3data
export P3_RESULTS="$P3_ROOT/results"
mkdir -p "$P3_RESULTS"
qstat -Qf batch_cpu > "$P3_RESULTS/batch_cpu.txt" 2>&1
qstat -Qf auto_free > "$P3_RESULTS/auto_free.txt" 2>&1
qstat -Bf > "$P3_RESULTS/server_limits.txt" 2>&1
bash scripts/build.sh
```

Inspect queue ACLs, enabled/started state, resource minima/maxima/defaults,
walltime, per-user and project limits. Queue visibility does not prove submission
permission. Confirm the course allocation with Prof. Chen if batch_cpu denies access.
Do not assume the queue limit is 720 cores just because the hardware supports it.

Use existing graphs where possible. See RUN.md for data preparation; the full
Friendster and R-MAT suite is large. The build needs GCC, OpenMPI, OpenMP and CUDA
for GPU binaries. Python analysis needs NumPy, SciPy and NetworkX.

## Run

```bash
# After all required data exist:
CPU_NODES=2 bash scripts/submit.sh cpu
bash scripts/submit.sh gpu
# Only after larger allocations are confirmed:
CPU_NODES="2 4 6 8 10" bash scripts/submit.sh cpu
# Optionally select an authorized queue:
P3_QUEUE=batch_cpu CPU_NODES=2 bash scripts/submit.sh cpu
```

The default CPU suite uses 1 and 2 nodes (up to 144 active cores). One-node
allocations reserve 72 cores even when a measurement uses only 1, 2, etc.;
speedup efficiency uses active ranks × OpenMP threads, not reserved capacity.
The 4-node design comparison runs only if that configuration is selected.
Do not submit both examples to the same results directory: CSV files append.
Each job must finish successfully; inspect its log and exit status before analysis.
GPU launches now require an explicit per-node PBS GPU assignment and reject
more ranks than visible GPUs. If mapping fails, inspect the site's scheduler GPU
environment; do not bypass the check by selecting arbitrary devices.

Run the small tests inside an appropriate CPU allocation:

```bash
python3 tools/smoke.py --serial bin/pagerank_serial --mpi bin/pagerank_mpi_omp
```

After CPU and GPU validation jobs finish, from the jobs directory:

```bash
cd "$P3_ROOT/jobs"
qsub -v P3_ROOT="$P3_ROOT",P3_DATA="$P3_DATA",P3_RESULTS="$P3_RESULTS" post.pbs
```

## Timing and interpretation

- `t_total`: maximum rank elapsed time for the iteration loop. Excludes graph
  loading, communication-plan setup, initial uploads, final output and warm-up.
- `t_iter = t_total / iters`; strong scaling uses identical graphs, iteration
  counts and damping, with `tol=0`. Compare medians over repetitions.
- `t_comp`: local computation, including ghost-buffer packing; `t_comm`: MPI
  calls, including waiting; `t_stage`: MPI+CUDA host/device staging, including
  scalar transfers. Components are separately maximized over ranks. They need
  not sum to `t_total`, so `comm_component_ratio` is not a measured wall-time share.
- Strong speedup `S(p)=T(1)/T(p)`; efficiency `S(p)/p`. The shared one-core
  MPI+OpenMP baseline is distinct from the serial-program baseline.
- Weak efficiency `T(1)/T(p)` is emitted only when a one-core measurement exists.
  Realized edges per active core means stored directed edges after removal of
  duplicates/self-loops, divided by active CPU cores. R-MAT's nominal work is
  2,097,152 candidate edges/core; realized work varies. Report M/p and N/p, and
  describe this as approximate weak scaling. Non-power-of-two graph sizes also
  change the R-MAT sampling distribution through rejection sampling.
- GPU scaling uses one MPI rank/device; efficiency is speedup/device count.
  GPU CSV `cores` is a legacy field, not the scheduler's reserved CPU count.
- `gbps` is an algorithmic traffic estimate, not measured DRAM bandwidth.
  Investigate the modest GPU speedup using matched loop timing, kernel variants,
  staging cost, degree skew and hardware details; do not claim a bottleneck
  without evidence. CPU/GPU jobs use the same Friendster iteration
  count (10); CPU uses 3 measured repetitions and GPU uses 5.

## Correctness and report

Validation compares full vectors with the serial program and NetworkX, checks
finite values, length, L1/L-infinity error and mass. It exits nonzero on numerical
failure. Verify that every expected CPU and GPU vector exists: a successful
comparison alone does not prove all variants were run. The small smoke suite
also exercises dangling/isolated vertices, no edges and empty rank partitions.

The report still needs measured Vanda figures, uncertainty/repetition counts,
hardware/compiler/job metadata, definitions, group names/IDs and data findings.
Record failed/denied submissions and queue waits separately from execution time.

AI disclosure suggestion: "OpenAI Codex assisted with code review, timing and
submission-script corrections, validation checks and experimental documentation.
The group reviewed these changes and verified reported measurements on Vanda."
Use the last sentence only after actually completing that verification.
