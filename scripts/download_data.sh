#!/bin/bash
# Download the SNAP datasets into $P3_DATA (run on the login node; compute nodes may lack internet).
# com-Friendster (large strong-scaling graph) is ~8.7 GB compressed; it stays gzipped and prep.pbs streams it.
set -e
source "$(dirname "$0")/env.sh"
mkdir -p "$P3_DATA" && cd "$P3_DATA"
for name in web-Google soc-LiveJournal1; do
  if [ ! -f "$name.txt" ]; then
    wget -q "https://snap.stanford.edu/data/$name.txt.gz" -O "$name.txt.gz"
    gunzip -f "$name.txt.gz"
  fi
done
if [ ! -f com-friendster.ungraph.txt.gz ]; then
  wget -c --progress=dot:giga "https://snap.stanford.edu/data/bigdata/communities/com-friendster.ungraph.txt.gz" \
    -O com-friendster.ungraph.txt.gz.part
  mv com-friendster.ungraph.txt.gz.part com-friendster.ungraph.txt.gz
fi
ls -lh "$P3_DATA"
