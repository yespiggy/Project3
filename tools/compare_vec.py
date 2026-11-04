"""Validation at scale: compare PageRank vectors written by --out against a reference vector.

Used for LiveJournal after exactly 50 iterations (tol=0): the serial program and the MPI+OpenMP program at
144 ... 720 ranks perform the same synchronous iteration, so the vectors must agree to rounding error
(summation order differs between partitions). Vectors are raw little-endian float64 arrays.

Usage: python3 compare_vec.py ref.bin out.json other1.bin [other2.bin ...]
Exits non-zero if any vector fails the thresholds below.
"""
import json
import sys

import numpy as np

L1_TOL, LINF_TOL, SUM_TOL = 1e-10, 1e-12, 1e-10


def main():
    ref_path, out_path, others = sys.argv[1], sys.argv[2], sys.argv[3:]
    ref = np.fromfile(ref_path, dtype="<f8")
    top_ref = np.argsort(-ref, kind="stable")[:100]
    res = {"reference": ref_path, "N": int(ref.size), "tolerances": {"l1": L1_TOL, "linf": LINF_TOL, "sum": SUM_TOL},
           "vectors": {}}
    ok = True
    for p in others:
        x = np.fromfile(p, dtype="<f8")
        if x.size != ref.size or not np.all(np.isfinite(x)):
            res["vectors"][p] = {"error": "size mismatch or non-finite values", "size": int(x.size)}
            ok = False
            continue
        d = np.abs(x - ref)
        top = np.argsort(-x, kind="stable")[:100]
        r = {"l1": float(d.sum()), "linf": float(d.max()), "sum": float(x.sum()),
             "top100_same_order": bool(np.array_equal(top, top_ref))}
        r["passed"] = r["l1"] <= L1_TOL and r["linf"] <= LINF_TOL and abs(r["sum"] - 1) <= SUM_TOL
        ok &= r["passed"]
        res["vectors"][p] = r
    res["passed"] = ok
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))
    if not ok:
        raise SystemExit("validation at scale FAILED")


if __name__ == "__main__":
    main()
