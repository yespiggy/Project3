#!/bin/bash
# mpirun wrapper for GPU ranks: restore the GPU(s) PBS assigned to this node (recorded by gpu_map in run_lib.sh),
# so a rank started over ssh on a second node does not pick another job's GPU.
# On Vanda PBS does not export CUDA_VISIBLE_DEVICES (the map reads "(not set)"); GPU jobs are then given whole
# GPUs by ngpus=, and the program selects device = local rank (it aborts if there are more local ranks than GPUs).
# The selected device and its bus id are printed by every rank, so the assignment can be checked in the job log.
f=${P3_GPUMAP:-/nonexistent}/$(hostname -s)
if [ -s "$f" ] && [ -n "$(cat "$f")" ]; then
  export CUDA_VISIBLE_DEVICES=$(cat "$f")
else
  echo "WARNING: no PBS GPU assignment recorded for $(hostname -s); using device = local rank" >&2
fi
exec "$@"
