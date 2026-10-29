"""Aggregate raw experiment CSVs into results/summary.json (medians over the 5 repetitions).

Usage: python3 aggregate.py <results_dir>
"""
import csv
import glob
import json
import os
import statistics
import sys


def rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return [r for r in csv.DictReader(f) if r.get("exp") and not r["exp"].startswith("exp")]


def med(rs, key):
    vals = [float(r[key]) for r in rs if r.get(key) not in (None, "")]
    return statistics.median(vals) if vals else None


def group(rs, *keys):
    out = {}
    for r in rs:
        out.setdefault(tuple(r[k] for k in keys), []).append(r)
    return out


def summarize(rs):
    r0 = rs[0]
    t = med(rs, "t_iter")
    return {
        "graph": r0["graph"], "N": int(r0["N"]), "M": int(r0["M"]),
        "nodes": int(r0["nodes"]), "ranks": int(r0["ranks"]), "threads": int(r0["threads"]), "cores": int(r0["cores"]),
        "part": r0["part"], "comm": r0["comm"], "reps": len(rs), "iters": int(r0["iters"]),
        "t_iter_ms": t * 1e3,
        "t_iter_min_ms": min(float(r["t_iter"]) for r in rs) * 1e3,
        "t_iter_max_ms": max(float(r["t_iter"]) for r in rs) * 1e3,
        "t_comp_ms": med(rs, "t_comp") / int(r0["iters"]) * 1e3,
        "t_comm_ms": med(rs, "t_comm") / int(r0["iters"]) * 1e3,
        "imbalance": med(rs, "imbalance"),
        "comm_mb_iter": med(rs, "comm_bytes_iter") / 1e6,
        "gteps": int(r0["M"]) / t / 1e9,
        "gbps": med(rs, "gbps"),
        "t_load_s": med(rs, "t_load"),
        "t_h2d_s": med(rs, "t_h2d"),
        "sumr": med(rs, "sumr"),
    }


def ncu_metrics(results):
    """Per-kernel Nsight Compute metrics (SpMV kernels only), if profiling succeeded."""
    out = {}
    for path in glob.glob(os.path.join(results, "ncu_*.csv")):
        kname = os.path.basename(path)[4:-4]
        with open(path, newline="") as f:
            lines = [l for l in f if l.startswith('"')]
        if not lines:
            continue
        per_kernel = {}
        for r in csv.DictReader(lines):
            name = r.get("Kernel Name", "")
            if "contrib" in name:
                continue
            try:
                val = float(r["Metric Value"].replace(",", ""))
            except (KeyError, ValueError):
                continue
            per_kernel.setdefault(name.split("(")[0], {})[r["Metric Name"]] = val
        out[kname] = per_kernel
    return out


def main():
    res = sys.argv[1]
    serial = rows(os.path.join(res, "e5_serial.csv"))
    t_serial = med(serial, "t_iter")
    s = {"serial": summarize(serial) if serial else None}

    e1 = {}
    e1_rows = rows(os.path.join(res, "e1.csv")) + rows(os.path.join(res, "e1_multi.csv"))
    for (exp, cores), rs in group(e1_rows, "exp", "cores").items():
        e1.setdefault(exp.replace("E1_", ""), []).append(summarize(rs))
    for mode, lst in e1.items():
        lst.sort(key=lambda x: x["cores"])
        base = lst[0]["t_iter_ms"]
        for x in lst:
            x["speedup_vs_1core"] = base / x["t_iter_ms"]
            x["efficiency"] = x["speedup_vs_1core"] / x["cores"]
            if t_serial:
                x["speedup_vs_serial"] = t_serial * 1e3 / x["t_iter_ms"]
    s["e1"] = e1

    e2_rows = rows(os.path.join(res, "e2.csv")) + rows(os.path.join(res, "e2_multi.csv"))
    e2 = [summarize(rs) for rs in group(e2_rows, "cores").values()]
    e2.sort(key=lambda x: x["cores"])
    for x in e2:
        x["weak_efficiency"] = e2[0]["t_iter_ms"] / x["t_iter_ms"]
        x["edges_per_core_m"] = x["M"] / x["cores"] / 1e6
    s["e2"] = e2

    s["e3"] = [summarize(rs) for rs in group(rows(os.path.join(res, "e3.csv")), "part", "comm").values()]

    e4 = []
    for (impl,), rs in group(rows(os.path.join(res, "e4.csv")), "impl").items():
        x = summarize(rs)
        x["kernel"] = impl.replace("cuda_", "")
        if t_serial:
            x["speedup_vs_serial"] = t_serial * 1e3 / x["t_iter_ms"]
        e4.append(x)
    s["e4"] = e4
    s["ncu"] = ncu_metrics(res)

    # E5: same graph, four platforms.
    e5 = []
    if serial:
        e5.append({"label": "串行 1 核", **s["serial"]})
    hyb = {x["cores"]: x for x in e1.get("hybrid", [])}
    mpi = {x["cores"]: x for x in e1.get("mpi", [])}
    if 16 in hyb or 16 in mpi:
        best16 = min([v for v in (hyb.get(16), mpi.get(16)) if v], key=lambda x: x["t_iter_ms"])
        e5.append({"label": "1 个 CPU 节点（16 核）", **best16})
    top = max(list(hyb) + list(mpi), default=None)
    if top and top > 16:
        best_top = min([v for v in (hyb.get(top), mpi.get(top)) if v], key=lambda x: x["t_iter_ms"])
        e5.append({"label": f"{best_top['nodes']} 个 CPU 节点（{top} 核）", **best_top})
    if e4:
        bestg = min(e4, key=lambda x: x["t_iter_ms"])
        e5.append({"label": f"1× A40（{bestg['kernel']}）", **bestg})
    for x in e5:
        if t_serial:
            x["speedup_vs_serial"] = t_serial * 1e3 / x["t_iter_ms"]
    s["e5"] = e5

    for name in ("validation", "google_analysis"):
        p = os.path.join(res, "val", f"{name}.json")
        if os.path.exists(p):
            with open(p) as f:
                s[name] = json.load(f)
    s["val_runs"] = [
        {k: r[k] for k in ("impl", "part", "comm", "ranks", "threads", "iters", "resid", "sumr")}
        for r in rows(os.path.join(res, "val", "val_cpu.csv")) + rows(os.path.join(res, "val", "val_gpu.csv"))
    ]
    with open(os.path.join(res, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2, ensure_ascii=False)
    print("wrote", os.path.join(res, "summary.json"))


if __name__ == "__main__":
    main()
