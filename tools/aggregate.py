"""Aggregate raw experiment CSVs into <results_dir>/summary.json (medians over the repetitions).

Usage: python3 aggregate.py <results_dir>

Understands both the file names of the first suite (serial.csv, strong_n*_lj.csv, ...) and those of the final suite
(jobs/final: serial_size.csv, strong_{lj,fr}_n*.csv, weak_n*.csv with E2_hybrid / E2_mpi, design_n*.csv, e4.csv with two
graphs, sweep_*, split_*, transport_*, multigpu_*, val/val_gpu*.csv, valscale/validation_scale.json, ...).
Rows are always grouped by everything that identifies a configuration (graph, experiment tag, cores, ...) so that two
graphs or two modes in one file are never pooled into one median.
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
        # a repeated header line (a CSV that was appended to without --header handling) is skipped, not parsed
        return [r for r in csv.DictReader(f) if r.get("exp") and r["exp"] != "exp"]


def med(rs, key):
    vals = [float(r[key]) for r in rs if r.get(key) not in (None, "")]
    return statistics.median(vals) if vals else None


def group(rs, *keys):
    out = {}
    for r in rs:
        out.setdefault(tuple(r[k] for k in keys), []).append(r)
    return out


def ratio(a, b):
    return a / (a + b) if a is not None and b is not None and a + b > 0 else None


def per_iter_ms(rs, key, iters):
    v = med(rs, key)
    return v / iters * 1e3 if v is not None and iters else None


def summarize(rs):
    r0 = rs[0]
    t = med(rs, "t_iter")
    iters = int(r0["iters"])
    x = {
        "exp": r0["exp"],
        "graph": r0["graph"], "N": int(r0["N"]), "M": int(r0["M"]),
        "nodes": int(r0["nodes"]), "ranks": int(r0["ranks"]), "threads": int(r0["threads"]), "cores": int(r0["cores"]),
        "part": r0["part"], "comm": r0["comm"], "reps": len(rs), "iters": iters,
        "t_iter_ms": t * 1e3,
        "t_iter_min_ms": min(float(r["t_iter"]) for r in rs) * 1e3,
        "t_iter_max_ms": max(float(r["t_iter"]) for r in rs) * 1e3,
        "t_comp_ms": per_iter_ms(rs, "t_comp", iters),
        "t_comm_ms": per_iter_ms(rs, "t_comm", iters),
        "imbalance": med(rs, "imbalance"),
        "comm_mb_iter": (med(rs, "comm_bytes_iter") or 0) / 1e6,
        "gteps": int(r0["M"]) / t / 1e9,
        "gbps": med(rs, "gbps"),
        "t_load_s": med(rs, "t_load"),
        "t_h2d_s": med(rs, "t_h2d"),
        "edges_per_core_m": int(r0["M"]) / int(r0["cores"]) / 1e6,
        "sumr": med(rs, "sumr"),
    }
    if med(rs, "t_wait") is not None:  # --split runs: barrier wait before each communication call
        x["t_wait_ms"] = per_iter_ms(rs, "t_wait", iters)
    return x


def files(res, pattern):
    out = []
    for path in sorted(glob.glob(os.path.join(res, pattern))):
        out += rows(path)
    return out


def read_csv_plain(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    res = sys.argv[1]

    # Serial baseline per graph: serial.csv (first suite) and the S_real rows of serial_size.csv (final suite).
    serial = rows(os.path.join(res, "serial.csv")) + [r for r in rows(os.path.join(res, "serial_size.csv"))
                                                      if r["exp"] == "S_real"]
    serial_by_graph = {k[0]: v for k, v in group(serial, "graph").items()}

    def t_serial_ms(graph):
        rs = serial_by_graph.get(graph)
        return med(rs, "t_iter") * 1e3 if rs else None

    s = {"serial": {g: summarize(rs) for g, rs in serial_by_graph.items()}}
    # serial run time against problem size (E5-style size scaling): R-MAT family, edge-factor sweep, real graphs
    s["serial_size"] = sorted((summarize(rs) for rs in group(rows(os.path.join(res, "serial_size.csv")),
                                                           "exp", "graph").values()),
                              key=lambda x: (x["exp"], x["M"]))

    # E1 strong scaling, per graph and mode. Speedup and efficiency are relative to the 1-core run of the same
    # graph (one rank, one thread; the same run serves both modes).
    e1 = {}
    for (graph, exp, cores), rs in group(files(res, "strong_*.csv"), "graph", "exp", "cores").items():
        e1.setdefault(graph, {}).setdefault(exp.replace("E1_", ""), []).append(summarize(rs))
    for graph, modes in e1.items():
        one = [x for lst in modes.values() for x in lst if x["cores"] == 1]
        base = one[0]["t_iter_ms"] if one else None
        t_ser = t_serial_ms(graph)
        for mode, lst in modes.items():
            lst.sort(key=lambda x: x["cores"])
            for x in lst:
                if base:
                    x["speedup"] = base / x["t_iter_ms"]
                    x["efficiency"] = x["speedup"] / x["cores"]
                x["comm_component_ratio"] = ratio(x["t_comm_ms"], x["t_comp_ms"])
                if t_ser:
                    x["speedup_vs_serial"] = t_ser / x["t_iter_ms"]
        if base and 1 not in [x["cores"] for x in modes.get("mpi", [])] and "mpi" in modes:
            modes["mpi"].insert(0, dict(one[0], speedup=1.0, efficiency=1.0))
    s["e1"] = e1

    # E2 weak scaling, per mode (E2_hybrid / E2_mpi; the first suite has the single tag E2). The 1-core run exists
    # for the hybrid mode only (identical to pure MPI), so it is the baseline of both modes.
    e2 = []
    for (exp, cores), rs in group(files(res, "weak_*.csv"), "exp", "cores").items():
        e2.append(summarize(rs))
    e2.sort(key=lambda x: (x["exp"], x["cores"]))
    weak_base = next((x for x in e2 if x["cores"] == 1), None)
    for x in e2:
        if weak_base:
            x["weak_efficiency"] = weak_base["t_iter_ms"] / x["t_iter_ms"]
        x["comm_component_ratio"] = ratio(x["t_comm_ms"], x["t_comp_ms"])
    s["e2"] = e2

    # E3 design comparison (partitioning x communication) at 288 ranks, per graph
    s["e3"] = [summarize(rs) for rs in group(files(res, "design_n*.csv"), "graph", "part", "comm").values()]

    # E4 single-GPU kernels, per graph (e4.csv holds LiveJournal and Friendster)
    e4 = []
    for (graph, impl), rs in group(rows(os.path.join(res, "e4.csv")), "graph", "impl").items():
        x = summarize(rs)
        x["kernel"] = impl.replace("cuda_", "")
        x["pct_of_peak_bw"] = x["gbps"] / A40_PEAK_GBPS
        t_ser = t_serial_ms(graph)
        if t_ser:
            x["speedup_vs_serial"] = t_ser / x["t_iter_ms"]
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
            x["t_stage_ms"] = per_iter_ms(rs, "t_stage", x["iters"])
            e6.append(x)
    for x in e6:
        one = [y for y in e6 if y["source"] == x["source"] and y["graph"] == x["graph"] and y["kernel"] == x["kernel"]
               and y["ranks"] == 1]
        if one:
            x["speedup_vs_1gpu"] = one[0]["t_iter_ms"] / x["t_iter_ms"]
            x["gpu_efficiency"] = x["speedup_vs_1gpu"] / x["ranks"]
    s["e6"] = e6

    # E5: CPU vs GPU per graph: serial, 1 core, 1 node (72 cores), 10 nodes (720 cores), best 1-GPU and 2-GPU runs.
    e5 = {}
    for graph, modes in e1.items():
        allrun = [x for lst in modes.values() for x in lst]
        lst = []
        if graph in serial_by_graph:
            lst.append({"label": "serial, 1 core", **s["serial"][graph]})
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

    # E7 ranks x threads at a fixed core count; E8 --split breakdown (t_wait); E9 transport comparison
    s["e7"] = sorted((summarize(rs) for rs in group(files(res, "sweep_*.csv"), "graph", "nodes", "ranks").values()),
                     key=lambda x: (x["graph"], x["nodes"], x["ranks"]))
    s["e8"] = sorted((summarize(rs) for rs in group(files(res, "split_*.csv"), "graph", "exp", "cores").values()),
                     key=lambda x: (x["graph"], x["cores"], x["exp"]))
    s["e9"] = sorted((summarize(rs) for rs in group(files(res, "transport_*.csv"), "graph", "exp", "cores").values()),
                     key=lambda x: (x["graph"], x["cores"], x["exp"]))

    # bandwidth probes (CPU STREAM at 1..72 threads, GPU copy / triad / random gather)
    s["stream_cpu"] = read_csv_plain(os.path.join(res, "stream_cpu.csv"))
    s["gpu_bw"] = read_csv_plain(os.path.join(res, "gpu_bw.csv"))

    # validation: web-Google (tools/validate.py), LiveJournal at scale (tools/compare_vec.py), run logs
    for name in ("validation", "google_analysis"):
        p = os.path.join(res, "val", f"{name}.json")
        if os.path.exists(p):
            with open(p) as f:
                s[name] = json.load(f)
    p = os.path.join(res, "valscale", "validation_scale.json")
    if os.path.exists(p):
        with open(p) as f:
            s["validation_scale"] = json.load(f)
    val_files = [os.path.join(res, "val", "val_cpu.csv")] + sorted(glob.glob(os.path.join(res, "val", "val_gpu*.csv")))
    val_files += sorted(glob.glob(os.path.join(res, "valscale", "valscale*.csv")))
    s["val_runs"] = [
        {k: r.get(k) for k in ("exp", "impl", "graph", "part", "comm", "ranks", "threads", "iters", "resid", "sumr")}
        for path in val_files for r in rows(path)
    ]

    # runs that failed inside a job (run_mpi logs them and carries on): they must be reported, never dropped
    fp = os.path.join(res, "failures.txt")
    s["failures"] = open(fp).read().splitlines() if os.path.exists(fp) else []

    with open(os.path.join(res, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2, ensure_ascii=False)
    print("wrote", os.path.join(res, "summary.json"))
    if s["failures"]:
        print(f"WARNING: {len(s['failures'])} failed run(s) listed in summary.json under 'failures'")


if __name__ == "__main__":
    main()
