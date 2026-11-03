#!/bin/bash
# mpirun wrapper for GPU ranks: restore the GPU(s) PBS assigned to this node (recorded by gpu_map in run_lib.sh),
# so a rank started over ssh on a second node does not pick another job's GPU.
f=${P3_GPUMAP:-/nonexistent}/$(hostname -s)
if [ ! -s "$f" ] || [ -z "$(cat "$f")" ]; then
  echo "ERROR: missing PBS GPU assignment for $(hostname -s)" >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES=$(cat "$f")
exec "$@"

