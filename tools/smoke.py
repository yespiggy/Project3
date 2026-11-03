"""Small deterministic correctness tests; run with --serial PATH [--mpi PATH].

MPI tests require mpirun and an allocation permitting four ranks.
Includes dangling vertices, isolated vertices, and empty edge partitions.
"""
import argparse
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import numpy as np
import networkx as nx

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--serial', required=True)
    parser.add_argument('--mpi')
    args = parser.parse_args()
    cases = {'cycle': (4, [(0, 1), (1, 2), (2, 3), (3, 0)]),
             'dangling': (6, [(0, 1), (0, 2), (1, 2), (2, 0), (3, 2)]),
             'edgeless': (4, []), 'sparse': (3, [(0, 1)])}
    count = 0
    with tempfile.TemporaryDirectory() as tmp:
        for name, (n, edges) in cases.items():
            graph = nx.DiGraph()
            graph.add_nodes_from(range(n))
            graph.add_edges_from(edges)
            ref_dict = nx.pagerank(graph, tol=1e-14 / n, max_iter=1000)
            ref = np.array([ref_dict[i] for i in range(n)])
            incoming = [[u for u, v in edges if v == row] for row in range(n)]
            off = [0]
            for row in incoming:
                off.append(off[-1] + len(row))
            path = Path(tmp) / (name + '.bin')
            with path.open('wb') as f:
                f.write(b'PRGRAPH1' + struct.pack('=qq', n, len(edges)))
                f.write(np.array(off, dtype=np.int64).tobytes())
                f.write(np.array([u for row in incoming for u in row], dtype=np.int32).tobytes())
                f.write(np.array([graph.out_degree(i) for i in range(n)], dtype=np.int32).tobytes())
            commands = [[args.serial]]
            if args.mpi:
                commands += [['mpirun', '-np', str(p), args.mpi, '--part', part, '--comm', comm]
                             for p in (1, 2, 4) for part in ('vertex', 'edge')
                             for comm in ('allgather', 'ghost')]
            for command in commands:
                out = Path(tmp) / 'rank.bin'
                subprocess.run(command + ['--graph', str(path), '--iters', '1000', '--tol', '1e-12',
                                          '--out', str(out)], check=True, stdout=subprocess.DEVNULL,
                               env={**os.environ, 'OMP_NUM_THREADS': '1'})
                actual = np.fromfile(out, dtype=np.float64)
                assert actual.shape == ref.shape and np.isfinite(actual).all()
                assert np.abs(actual-ref).sum() < 1e-10, (name, command)
                assert abs(actual.sum()-1) < 1e-12 and (actual >= 0).all()
                count += 1
    print(f'PASS: {count} PageRank comparisons against NetworkX')

if __name__ == '__main__':
    main()
