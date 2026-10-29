"""Aggregate raw experiment CSVs into results/summary.json (medians over the 5 repetitions).

Usage: python3 aggregate.py <results_dir>
"""
import csv
import glob
import json
import os
import statistics
import sys

A40_PEAK_GBPS = 696.0  # A40 theoretical DRAM bandwidth


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
        "edges_per_core_m": int(r0["M"]) / int(r0["cores"]) / 1e6,
        "sumr": med(rs, "sumr"),
    }


def files(res, pattern):
    out = []
    for path in sorted(glob.glob(os.path.join(res, pattern))):
        out += rows(path)
    return out


def main():
    res = sys.argv[1]
    serial = rows(os.path.join(res, "serial.csv"))
    t_serial = med(serial, "t_iter")
    s = {"serial": summarize(serial) if serial else None}

    # E1 strong scaling, per graph and mode. Speedup and efficiency are relative to the 1-core run of the same
    # graph (one rank, one thread; the same run serves both modes).
    e1 = {}
    for (graph, exp, cores), rs in group(files(res, "strong_*.csv"), "graph", "exp", "cores").items():
        e1.setdefault(graph, {}).setdefault(exp.replace("E1_", ""), []).append(summarize(rs))
    for graph, modes in e1.items():
        one = [x for lst in modes.values() for x in lst if x["cores"] == 1]
        base = one[0]["t_iter_ms"] if one else None
        for mode, lst in modes.items():
            lst.sort(key=lambda x: x["cores"])
            for x in lst:
                if base:
                    x["speedup"] = base / x["t_iter_ms"]
                    x["efficiency"] = x["speedup"] / x["cores"]
                x["comm_fraction"] = x["t_comm_ms"] / (x["t_comp_ms"] + x["t_comm_ms"])
                if t_serial and graph == serial[0]["graph"]:
                    x["speedup_vs_serial"] = t_serial * 1e3 / x["t_iter_ms"]
        if base and 1 not in [x["cores"] for x in modes.get("mpi", [])] and "mpi" in modes:
            modes["mpi"].insert(0, dict(one[0], speedup=1.0, efficiency=1.0))
    s["e1"] = e1

    # E2 weak scaling: realized edges per core = M / cores after duplicate and self-loop removal.
    e2 = [summarize(rs) for rs in group(files(res, "weak_*.csv"), "cores").values()]
    e2.sort(key=lambda x: x["cores"])
    for x in e2:
        x["weak_efficiency"] = e2[0]["t_iter_ms"] / x["t_iter_ms"]
        x["edges_per_core_m"] = x["M"] / x["cores"] / 1e6
        x["comm_fraction"] = x["t_comm_ms"] / (x["t_comp_ms"] + x["t_comm_ms"])
    s["e2"] = e2

    s["e3"] = [summarize(rs) for rs in group(rows(os.path.join(res, "design_n4.csv")), "part", "comm").values()]

    e4 = []
    for (impl,), rs in group(rows(os.path.join(res, "e4.csv")), "impl").items():
        x = summarize(rs)
        x["kernel"] = impl.replace("cuda_", "")
        x["pct_of_peak_bw"] = x["gbps"] / A40_PEAK_GBPS
        if t_serial:
            x["speedup_vs_serial"] = t_serial * 1e3 / x["t_iter_ms"]
        e4.append(x)
    s["e4"] = e4

    # E6 multi-GPU (MPI+CUDA): 1 GPU, 2 GPUs in one node, 2 GPUs on two nodes.
    e6 = []
    for name, where in (("multigpu_1node.csv", "same node"), ("multigpu_2node.csv", "two nodes")):
        for (graph, impl, ranks), rs in group(rows(os.path.join(res, name)), "graph", "impl", "ranks").items():
            x = summarize(rs)
            x["kernel"] = impl.replace("mpi_cuda_", "")
            x["config"] = "1 GPU" if x["ranks"] == 1 else f"{x['ranks']} GPUs, {where}"
            x["source"] = name
            x["t_stage_ms"] = med(rs, "t_stage") / x["iters"] * 1e3
            e6.append(x)
    for x in e6:
        one = [y for y in e6 if y["source"] == x["source"] and y["graph"] == x["graph"] and y["kernel"] == x["kernel"]
               and y["ranks"] == 1]
        if one:
            x["speedup_vs_1gpu"] = one[0]["t_iter_ms"] / x["t_iter_ms"]
    s["e6"] = e6

    # E5: CPU vs GPU per graph: 1 core, 1 node (72 cores), 10 nodes (720 cores), best 1-GPU and 2-GPU runs.
    e5 = {}
    for graph, modes in e1.items():
        allrun = [x for lst in modes.values() for x in lst]
        lst = []
        if serial and graph == serial[0]["graph"]:
            lst.append({"label": "serial, 1 core", **s["serial"]})
        for c, label in ((1, "1 core"), (72, "1 CPU node (72 cores)"), (720, "10 CPU nodes (720 cores)")):
            cand = [x for x in allrun if x["cores"] == c]
            if cand:
                lst.append({"label": label, **min(cand, key=lambda x: x["t_iter_ms"])})
        gpus = [x for x in e6 if x["graph"] == graph]
        for n in (1, 2):
            cand = [x for x in gpus if x["ranks"] == n]
            if cand:
                b = min(cand, key=lambda x: x["t_iter_ms"])
                lst.append({"label": f"{b['config']} ({b['kernel']})", **b})
        e4b = [x for x in e4 if x["graph"] == graph]
        if e4b:
            b = min(e4b, key=lambda x: x["t_iter_ms"])
            lst.append({"label": f"1 GPU, single-GPU program ({b['kernel']})", **b})
        base = lst[0]["t_iter_ms"] if lst else None
        for x in lst:
            x["speedup_vs_first"] = base / x["t_iter_ms"]
        e5[graph] = lst
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
