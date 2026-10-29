#!/bin/bash
# Download the SNAP datasets into $P3_DATA (run on the login node; compute nodes may lack internet).
set -e
source "$(dirname "$0")/env.sh"
mkdir -p "$P3_DATA" && cd "$P3_DATA"
for name in web-Google soc-LiveJournal1; do
  if [ ! -f "$name.txt" ]; then
    wget -q "https://snap.stanford.edu/data/$name.txt.gz" -O "$name.txt.gz"
    gunzip -f "$name.txt.gz"
  fi
done
ls -lh "$P3_DATA"
