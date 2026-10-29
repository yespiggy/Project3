"""Data findings for web-Google from the converged serial PageRank vector.

Usage: python3 analyze_google.py <web-Google.bin> <web-Google.idmap> <val_dir> -> <val_dir>/google_analysis.json
"""
import json
import os
import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr

from validate import load_graph


def main():
    gpath, idmap_path, vdir = sys.argv[1:4]
    n, m, off, src, outdeg = load_graph(gpath)
    indeg = np.diff(off)
    orig = np.fromfile(idmap_path, dtype=np.int64)
    r = np.fromfile(os.path.join(vdir, "serial.bin"), dtype=np.float64)
    order = np.argsort(-r)

    top = [
        {"rank": i + 1, "node": int(orig[v]), "pagerank": float(r[v]), "indeg": int(indeg[v]), "outdeg": int(outdeg[v]),
         "indeg_rank": int((indeg > indeg[v]).sum() + 1)}
        for i, v in enumerate(order[:15])
    ]
    share = {f"top_{p}pct": float(r[order[: max(1, int(n * p / 100))]].sum()) for p in (0.1, 1, 10)}

    # In-degree distribution on log2 bins: [0], [1], [2,3], [4,7], ...
    bins = [0, 1]
    while bins[-1] <= indeg.max():
        bins.append(bins[-1] * 2)
    hist, _ = np.histogram(indeg, bins=bins + [bins[-1] * 2])
    deg_hist = [{"lo": int(bins[i]), "hi": int(bins[i + 1] - 1), "count": int(c)} for i, c in enumerate(hist) if c > 0]

    # Mean PageRank per in-degree bin, to show how strongly rank follows in-degree.
    pr_by_deg = []
    for i in range(len(bins) - 1):
        mask = (indeg >= bins[i]) & (indeg < bins[i + 1])
        if mask.any():
            pr_by_deg.append({"lo": int(bins[i]), "hi": int(bins[i + 1] - 1), "mean_pr": float(r[mask].mean()),
                              "count": int(mask.sum())})

    resid = np.loadtxt(os.path.join(vdir, "serial_resid.csv"), delimiter=",", skiprows=1)
    top100_by_pr, top100_by_deg = set(order[:100]), set(np.argsort(-indeg)[:100])
    out = {
        "N": n, "M": m,
        "dangling": int((outdeg == 0).sum()),
        "zero_indeg": int((indeg == 0).sum()),
        "max_indeg": int(indeg.max()), "max_outdeg": int(outdeg.max()),
        "mean_degree": m / n,
        "spearman_pr_indeg": float(spearmanr(r, indeg).statistic),
        "pearson_logpr_logindeg": float(pearsonr(np.log(r), np.log1p(indeg)).statistic),
        "top100_overlap_pr_vs_indeg": len(top100_by_pr & top100_by_deg),
        "rank_mass_share": share,
        "min_pr": float(r.min()), "max_pr": float(r.max()), "uniform_pr": 1.0 / n,
        "top": top,
        "indeg_hist": deg_hist,
        "pr_by_indeg": pr_by_deg,
        "convergence": [{"iter": int(i), "resid": float(v)} for i, v in resid],
    }
    with open(os.path.join(vdir, "google_analysis.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k not in ("indeg_hist", "pr_by_indeg", "convergence")}, indent=2))


if __name__ == "__main__":
    main()
