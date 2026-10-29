"""Validate every PageRank implementation on web-Google.

Compares each output vector against the serial C++ result, and the serial result against NetworkX.
Usage: python3 validate.py <web-Google.bin> <val_dir>   ->  writes <val_dir>/validation.json
"""
import glob
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp


def load_graph(path):
    with open(path, "rb") as f:
        assert f.read(8) == b"PRGRAPH1"
        n, m = np.frombuffer(f.read(16), dtype=np.int64)
        off = np.frombuffer(f.read(8 * (n + 1)), dtype=np.int64)
        src = np.frombuffer(f.read(4 * m), dtype=np.int32)
        outdeg = np.frombuffer(f.read(4 * n), dtype=np.int32)
    return int(n), int(m), off, src, outdeg


def compare(r, ref, k=100):
    top_r, top_ref = np.argsort(-r)[:k], np.argsort(-ref)[:k]
    top1k = len(set(np.argsort(-r)[:1000]) & set(np.argsort(-ref)[:1000]))
    return {
        "l1": float(np.abs(r - ref).sum()),
        "linf": float(np.abs(r - ref).max()),
        "sum": float(r.sum()),
        f"top{k}_same_order": bool((top_r == top_ref).all()),
        "top1000_overlap": top1k,
    }


def main():
    gpath, vdir = sys.argv[1], sys.argv[2]
    n, m, off, src, outdeg = load_graph(gpath)
    serial = np.fromfile(os.path.join(vdir, "serial.bin"), dtype=np.float64)
    out = {"graph": os.path.basename(gpath), "N": n, "M": m, "implementations": {}}

    # NetworkX reference. Edge u->v is stored in row v of the in-edge CSR, so the adjacency is the transpose.
    import networkx as nx

    t = time.time()
    dst = np.repeat(np.arange(n, dtype=np.int64), np.diff(off))
    adj = sp.csr_array((np.ones(m), (src.astype(np.int64), dst)), shape=(n, n))
    G = nx.from_scipy_sparse_array(adj, create_using=nx.DiGraph)
    pr = nx.pagerank(G, alpha=0.85, tol=1e-13, max_iter=1000)
    ref = np.array([pr[i] for i in range(n)])
    out["networkx"] = {"version": nx.__version__, "seconds": round(time.time() - t, 1), **compare(serial, ref)}

    for path in sorted(glob.glob(os.path.join(vdir, "*.bin"))):
        name = os.path.basename(path)[:-4]
        if name == "serial":
            continue
        r = np.fromfile(path, dtype=np.float64)
        out["implementations"][name] = compare(r, serial)

    with open(os.path.join(vdir, "validation.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
