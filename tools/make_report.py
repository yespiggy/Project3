"""Build every figure and table of the report from the raw CSV files in results_final/.

Usage: python3 tools/make_report.py [results_final] [report]
Writes report/figures/*.pdf and report/generated/*.tex (tables and \\newcommand numbers used in report.tex).
Every number in the report comes from here: medians over the timed repetitions of each configuration
(the warm-up repetition is never written to the CSV files).
"""
import csv
import glob
import json
import os
import statistics
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator, ScalarFormatter  # noqa: E402

RES = sys.argv[1] if len(sys.argv) > 1 else "results_final"
OUT = sys.argv[2] if len(sys.argv) > 2 else "report"
FIG, GEN = f"{OUT}/figures", f"{OUT}/generated"
os.makedirs(FIG, exist_ok=True)
os.makedirs(GEN, exist_ok=True)

# Categorical slots 1-3 of the reference palette (validated all-pairs); gray for reference lines.
C1, C2, C3, GRAY, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 8.5, "axes.titlelocation": "left", "axes.titleweight": "bold", "axes.labelsize": 8.5, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "lines.linewidth": 1.5,
    "lines.markersize": 4.5, "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight",
})
CORES = [1, 2, 4, 8, 16, 36, 72, 144, 288, 432, 576, 720]
TICKS = [1, 4, 16, 72, 144, 288, 720]


def rows(pattern):
    out = []
    for f in sorted(glob.glob(f"{RES}/{pattern}")):
        with open(f, newline="") as fh:
            out += [r for r in csv.DictReader(fh) if r.get("exp") and r["exp"] != "exp"]
    return out


def groups(rs, *keys):
    g = defaultdict(list)
    for r in rs:
        g[tuple(r[k] for k in keys)].append(r)
    return g


def med(rs, key, scale=1.0):
    v = [float(r[key]) for r in rs if r.get(key) not in (None, "")]
    return statistics.median(v) * scale if v else None


def per_iter(rs, key):
    """median of a run total (t_comp, t_comm, t_wait) divided by the iteration count, in ms"""
    v = [float(r[key]) / int(r["iters"]) * 1e3 for r in rs if r.get(key) not in (None, "")]
    return statistics.median(v) if v else None


def series(rs, exp, graph=None):
    """{cores: rows} for one experiment tag"""
    s = defaultdict(list)
    for r in rs:
        if r["exp"] == exp and (graph is None or r["graph"] == graph):
            s[int(r["cores"])].append(r)
    return dict(sorted(s.items()))


def logx_cores(ax, cores=TICKS):
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(cores))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([str(c) for c in cores], rotation=0)
    ax.set_xlabel("CPU cores")


def plain_y(ax):
    """decimal tick labels on a log axis with a small range (avoids 6x10^0 style labels)"""
    ax.yaxis.set_major_locator(FixedLocator([0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())


def save(fig, name):
    # Multi-panel figures: label the panels (a), (b) so captions can refer to them.
    titled = [ax for ax in fig.axes if ax.get_title(loc="left")]
    if len(titled) > 1:
        for i, ax in enumerate(titled):
            ax.set_title(f"({'abcd'[i]}) {ax.get_title(loc='left')}", loc="left")
    fig.savefig(f"{FIG}/{name}.pdf")
    plt.close(fig)


nums = {}  # \newcommand macros for the text


def num(name, value, fmt="{:.2f}"):
    nums[name] = fmt.format(value) if value is not None else "n/a"


def eff(x):
    """efficiency with two significant digits below 0.1 (0.0042), two decimals above"""
    return f"{x:.2f}" if x >= 0.095 else f"{x:.2g}"


def fmt_ms(x):
    if x is None:
        return "--"
    return f"{x:.3g}" if x < 100 else f"{x:.0f}"


# ---------------------------------------------------------------- serial scaling (S_*)
ser = rows("serial_size.csv")
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
spec = [("S_ef", "R-MAT, $N=2^{20}$ fixed, edge factor 4-64", C1, "o"),
        ("S_rmat", "R-MAT, $N$ and $M$ grow together", C2, "s"),
        ("S_real", "real graphs", C3, "^")]
serial_tab = []
for exp, label, c, mk in spec:
    g = groups([r for r in ser if r["exp"] == exp], "graph")
    pts = sorted(((int(v[0]["M"]), int(v[0]["N"]), med(v, "t_iter", 1e3), k[0]) for k, v in g.items()))
    M = [p[0] for p in pts]
    T = [p[2] for p in pts]
    nspe = [t * 1e6 / m for m, t in zip(M, T)]  # ns per edge
    axs[0].plot(M, T, marker=mk, color=c, label=label, linestyle="-" if exp != "S_real" else "none")
    axs[1].plot(M, nspe, marker=mk, color=c, label=label, linestyle="-" if exp != "S_real" else "none")
    for m, n, t, name in pts:
        serial_tab.append((exp, name, n, m, t, t * 1e6 / m))
lo, hi = 1.5e6, 5e9
ref = [x * 1e-6 * 2.0 for x in (lo, hi)]  # a line of slope 1 (time proportional to M) for reference
axs[0].plot([lo, hi], ref, color=GRAY, linestyle="--", linewidth=1, label="slope 1 ($t \\propto M$)")
for ax in axs:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("edges $M$")
axs[0].set_ylabel("time per iteration (ms)")
axs[1].set_ylabel("time per edge (ns)")
axs[0].set_title("Serial time per iteration")
axs[1].set_title("Serial time per edge")
axs[0].legend(loc="upper left", fontsize=6.5)
axs[1].set_yscale("linear")
axs[1].set_ylim(0, 8)
save(fig, "serial_scaling")
with open(f"{GEN}/tab_serial.tex", "w") as f:
    names = {"S_ef": "fixed $N$", "S_rmat": "$N,M$ grow", "S_real": "real"}
    for exp, name, n, m, t, ns in serial_tab:
        f.write(f"{names[exp]} & \\texttt{{{name.replace('_', '\\_')}}} & {n/1e6:.2f} & {m/1e6:.1f} & "
                f"{m/n:.1f} & {fmt_ms(t)} & {ns:.2f} \\\\\n")
real = {name: (t, ns) for exp, name, n, m, t, ns in serial_tab if exp == "S_real"}
num("serLJms", real.get("soc-LiveJournal1", (None,))[0], "{:.0f}")
num("serFRs", real.get("com-friendster", (None,))[0] / 1e3, "{:.1f}")
ef = sorted((m, ns) for exp, name, n, m, t, ns in serial_tab if exp == "S_ef")
rm = sorted((m, ns) for exp, name, n, m, t, ns in serial_tab if exp == "S_rmat")
num("serEFlo", ef[0][1]), num("serEFhi", ef[-1][1])
num("serRMlo", rm[0][1]), num("serRMhi", rm[-1][1])

# ---------------------------------------------------------------- strong scaling (E1)
strong = {"lj": rows("strong_lj_n*.csv"), "fr": rows("strong_fr_n*.csv")}
gname = {"lj": "soc-LiveJournal1", "fr": "com-friendster"}
title = {"lj": "soc-LiveJournal1 (4.8M vertices)", "fr": "com-friendster (65.6M vertices)"}
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7))
strong_tab = {}
for ax, key in zip(axs, ["lj", "fr"]):
    hy = series(strong[key], "E1_hybrid")
    mp = series(strong[key], "E1_mpi")
    t1 = med(hy[1], "t_iter", 1e3)
    for s, lab, c, mk in [(hy, "MPI+OpenMP (1 rank / NUMA domain)", C1, "o"), (mp, "pure MPI (1 rank / core)", C2, "s")]:
        xs = sorted(s)
        if s is mp and 1 not in s:
            xs = [1] + xs  # the 1-core run is the same program for both modes
        ys = [med(s[c_], "t_iter", 1e3) if c_ in s else t1 for c_ in xs]
        ax.plot(xs, ys, marker=mk, color=c, label=lab)
    ax.plot([1, 720], [t1, t1 / 720], color=GRAY, linestyle="--", linewidth=1, label="ideal ($T_1/p$)")
    ax.axvline(72, color=GRAY, linewidth=0.8, linestyle=":")
    ax.text(76, ax.get_ylim()[1] if False else t1, "1 node | multi-node", fontsize=6.5, color=INK2, va="top")
    logx_cores(ax)
    ax.set_yscale("log")
    ax.set_ylabel("time per iteration (ms)")
    ax.set_title(title[key])
    strong_tab[key] = {"t1": t1, "hy": {c_: med(v, "t_iter", 1e3) for c_, v in hy.items()},
                       "mp": {c_: med(v, "t_iter", 1e3) for c_, v in mp.items()},
                       "hyr": hy, "mpr": mp}
axs[0].legend(loc="lower left")
save(fig, "strong_time")

fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7))
for key, ls in [("lj", "-"), ("fr", "--")]:
    t1 = strong_tab[key]["t1"]
    for mode, c, mk, lab in [("hy", C1, "o", "MPI+OpenMP"), ("mp", C2, "s", "pure MPI")]:
        d = dict(strong_tab[key][mode])
        d.setdefault(1, t1)
        xs = sorted(d)
        sp = [t1 / d[x] for x in xs]
        axs[0].plot(xs, sp, marker=mk, color=c, linestyle=ls, label=f"{lab}, {gname[key]}",
                    markerfacecolor=c if key == "lj" else "white")
        axs[1].plot(xs, [s_ / x for s_, x in zip(sp, xs)], marker=mk, color=c, linestyle=ls,
                    markerfacecolor=c if key == "lj" else "white")
axs[0].plot([1, 720], [1, 720], color=GRAY, linestyle=":", linewidth=1, label="ideal")
for ax in axs:
    logx_cores(ax)
    ax.set_yscale("log")
axs[0].set_ylabel("speedup $S_p = T_1 / T_p$")
axs[1].set_ylabel("parallel efficiency $E_p = S_p / p$")
axs[0].set_title("Speedup")
axs[1].set_title("Parallel efficiency")
axs[0].legend(loc="upper left", fontsize=6.5)
save(fig, "strong_speedup")

with open(f"{GEN}/tab_strong.tex", "w") as f:
    for c_ in CORES:
        cells = [str(c_)]
        for key in ["lj", "fr"]:
            t1 = strong_tab[key]["t1"]
            for mode in ["hy", "mp"]:
                t = strong_tab[key][mode].get(c_, t1 if c_ == 1 else None)
                cells += [fmt_ms(t), f"{t1/t:.2g}" if t and t1 / t < 10 else (f"{t1/t:.0f}" if t else "--"),
                          eff(t1 / t / c_) if t else "--"]
        f.write(" & ".join(cells) + " \\\\\n")
for key in ["lj", "fr"]:
    K = key.upper()
    hy, mp, t1 = strong_tab[key]["hy"], strong_tab[key]["mp"], strong_tab[key]["t1"]
    best = min(hy, key=lambda c_: hy[c_])
    num(f"st{K}one", t1, "{:.1f}" if t1 < 1000 else "{:.0f}")
    num(f"st{K}best", hy[best], "{:.2f}" if hy[best] < 100 else "{:.0f}")
    num(f"st{K}bestc", best, "{:d}")
    num(f"st{K}bestS", t1 / hy[best], "{:.0f}")
    for c_ in [36, 72, 144, 288, 720]:
        if c_ in hy:
            num(f"st{K}hy{c_}", hy[c_], "{:.2f}" if hy[c_] < 100 else "{:.0f}")
            num(f"st{K}hyS{c_}", t1 / hy[c_], "{:.1f}")
            num(f"st{K}hyE{c_}", t1 / hy[c_] / c_, "{:.2f}")
        if c_ in mp:
            num(f"st{K}mp{c_}", mp[c_], "{:.2f}" if mp[c_] < 100 else "{:.0f}")
            num(f"st{K}mpS{c_}", t1 / mp[c_], "{:.1f}")
num("stLJratio", strong_tab["lj"]["mp"][432] / strong_tab["lj"]["hy"][432], "{:.0f}")
num("stFRratio", strong_tab["fr"]["mp"][720] / strong_tab["fr"]["hy"][720], "{:.0f}")

# ---------------------------------------------------------------- communication volume and comm/comp split
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
for key, ls in [("lj", "-"), ("fr", "--")]:
    for mode, c, mk, lab in [("hyr", C1, "o", "MPI+OpenMP"), ("mpr", C2, "s", "pure MPI")]:
        s = strong_tab[key][mode]
        xs = [x for x in sorted(s) if x > 1 and med(s[x], "comm_bytes_iter")]
        axs[0].plot(xs, [med(s[x], "comm_bytes_iter") / 1e6 for x in xs], marker=mk, color=c, linestyle=ls,
                    markerfacecolor=c if key == "lj" else "white", label=f"{lab}, {gname[key]}")
for key, ls in [("lj", "-"), ("fr", "--")]:
    for mode, c, mk, lab in [("hyr", C1, "o", "MPI+OpenMP"), ("mpr", C2, "s", "pure MPI")]:
        s = strong_tab[key][mode]
        xs = [x for x in sorted(s) if x > 1]
        fr_ = [per_iter(s[x], "t_comm") / (per_iter(s[x], "t_comm") + per_iter(s[x], "t_comp")) for x in xs]
        axs[1].plot(xs, fr_, marker=mk, color=c, linestyle=ls, markerfacecolor=c if key == "lj" else "white")
for ax in axs:
    logx_cores(ax, [2, 8, 36, 72, 144, 288, 720])
axs[0].set_yscale("log")
axs[0].set_ylabel("ghost data sent per iteration (MB)")
axs[0].set_title("Ghost data per iteration")
axs[1].set_ylabel("$t_{comm} / (t_{comm} + t_{comp})$")
axs[1].set_ylim(0, 1.02)
axs[1].set_title("Share of time in MPI calls")
fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=6.5,
           bbox_to_anchor=(0.5, -0.14))
save(fig, "comm_volume")

# --split breakdown: compute / wait (imbalance + synchronisation) / data movement, per iteration
split = {"lj": rows("split_lj_n*.csv"), "fr": rows("split_fr_n*.csv")}
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=False)
split_tab = {}
for ax, key in zip(axs, ["lj", "fr"]):
    s = series(split[key], "E8_hybrid")
    xs = sorted(s)
    comp = [per_iter(s[x], "t_comp") for x in xs]
    wait = [per_iter(s[x], "t_wait") or 0.0 for x in xs]
    comm = [per_iter(s[x], "t_comm") for x in xs]
    split_tab[key] = dict(zip(xs, zip(comp, wait, comm)))
    ax.plot(xs, comp, marker="o", color=C1, label="$t_{comp}$ (local work)")
    ax.plot(xs, wait, marker="^", color=C3, label="$t_{wait}$ (barrier: imbalance)")
    ax.plot(xs, comm, marker="s", color=C2, label="$t_{comm}$ (data exchange + Allreduce)")
    logx_cores(ax, [x for x in TICKS if x in xs])
    ax.set_yscale("log")
    ax.set_ylabel("ms per iteration")
    ax.set_title(gname[key])
    plain_y(ax)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.13))
save(fig, "split_breakdown")
for key in ["lj", "fr"]:
    K = key.upper()
    for c_ in [72, 144, 720]:
        if c_ in split_tab[key]:
            cp, wt, cm = split_tab[key][c_]
            num(f"sp{K}comp{c_}", cp, "{:.2f}")
            num(f"sp{K}wait{c_}", wt, "{:.2f}")
            num(f"sp{K}comm{c_}", cm, "{:.2f}")

# ---------------------------------------------------------------- weak scaling (E2)
weak = rows("weak_n*.csv")
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
weak_tab = {}
for mode, c, mk, lab in [("E2_hybrid", C1, "o", "MPI+OpenMP"), ("E2_mpi", C2, "s", "pure MPI")]:
    s = series(weak, mode)
    if mode == "E2_mpi":
        s = {1: series(weak, "E2_hybrid")[1], **s}
    xs = sorted(s)
    ts = [med(s[x], "t_iter", 1e3) for x in xs]
    weak_tab[mode] = dict(zip(xs, ts))
    axs[0].plot(xs, ts, marker=mk, color=c, label=lab)
    axs[1].plot(xs, [ts[0] / t for t in ts], marker=mk, color=c, label=lab)
t1w = weak_tab["E2_hybrid"][1]
axs[0].axhline(t1w, color=GRAY, linestyle="--", linewidth=1, label="ideal (constant)")
axs[1].axhline(1, color=GRAY, linestyle="--", linewidth=1)
for ax in axs:
    logx_cores(ax)
    ax.set_yscale("log")
axs[0].set_ylabel("time per iteration (ms)")
axs[1].set_ylabel("weak efficiency $T_1 / T_p$")
axs[0].set_title("Time per iteration")
axs[1].set_title("Weak-scaling efficiency")
axs[0].legend(loc="upper left")
save(fig, "weak")
with open(f"{GEN}/tab_weak.tex", "w") as f:
    wr = groups([r for r in weak if r["exp"] == "E2_hybrid"], "cores")
    for c_ in CORES:
        r0 = wr[(str(c_),)][0] if (str(c_),) in wr else None
        h = weak_tab["E2_hybrid"].get(c_)
        m = weak_tab["E2_mpi"].get(c_)
        f.write(f"{c_} & {int(r0['N'])/1e6:.2f} & {int(r0['M'])/1e6:.0f} & {fmt_ms(h)} & "
                f"{eff(t1w / h)} & {fmt_ms(m)} & {eff(t1w / m) if m else '--'} \\\\\n")
for c_ in [36, 72, 720]:
    num(f"wkHy{c_}", weak_tab["E2_hybrid"][c_], "{:.1f}")
    num(f"wkHyE{c_}", t1w / weak_tab["E2_hybrid"][c_], "{:.2f}")
num("wkOne", t1w, "{:.2f}")
num("wkMp288", weak_tab["E2_mpi"].get(288), "{:.0f}")
num("wkMp144", weak_tab["E2_mpi"].get(144), "{:.1f}")

# ---------------------------------------------------------------- memory bandwidth (STREAM vs PageRank)
st = rows("stream_cpu.csv") if False else list(csv.DictReader(open(f"{RES}/stream_cpu.csv")))
fig, ax = plt.subplots(figsize=(3.6, 2.6))
th = [int(r["threads"]) for r in st]
ax.plot(th, [float(r["triad_gbs"]) for r in st], marker="o", color=C1, label="STREAM triad")
hy1 = strong_tab["lj"]["hyr"]
xs = [x for x in sorted(hy1) if x <= 72]
ax.plot(xs, [med(hy1[x], "gbps") for x in xs], marker="s", color=C2, label="PageRank (soc-LiveJournal1)")
frh = strong_tab["fr"]["hyr"]
xs2 = [x for x in sorted(frh) if x <= 72]
ax.plot(xs2, [med(frh[x], "gbps") for x in xs2], marker="^", color=C3, label="PageRank (com-friendster)")
logx_cores(ax, [1, 2, 4, 8, 16, 36, 72])
ax.set_xlabel("cores (OpenMP threads) on one node")
ax.set_yscale("log")
ax.set_ylabel("GB/s")
ax.set_title("Memory bandwidth, one node")
ax.legend(loc="lower right")
save(fig, "bandwidth")
num("streamOne", float(st[0]["triad_gbs"]), "{:.0f}")
num("streamFull", float(st[-1]["triad_gbs"]), "{:.0f}")
num("streamHalf", float([r for r in st if r["threads"] == "36"][0]["triad_gbs"]), "{:.0f}")
num("prLJbw36", med(hy1[36], "gbps"), "{:.0f}")
num("prFRbw72", med(frh[72], "gbps"), "{:.0f}")

# ---------------------------------------------------------------- GPU
gbw = list(csv.DictReader(open(f"{RES}/gpu_bw.csv")))[0]
num("gpuCopy", float(gbw["copy_gbs"]), "{:.0f}")
num("gpuTriad", float(gbw["triad_gbs"]), "{:.0f}")
num("gpuGather", float(gbw["random_gather_gbs"]), "{:.0f}")
e4 = rows("e4.csv")
kern = groups(e4, "impl")
mg1 = rows("multigpu_1node.csv")
mg2 = rows("multigpu_2node.csv")
gpu_lj = {}
for impl in ["cuda_thread", "cuda_warp", "cuda_hybrid", "cuda_cusparse"]:
    gpu_lj[impl] = (med(kern[(impl,)], "t_iter", 1e3), med(kern[(impl,)], "gbps"))
lj_cpu = strong_tab["lj"]
bars = [("serial, 1 core", real["soc-LiveJournal1"][0]),
        ("CPU, 36 cores (1 socket)", lj_cpu["hy"][36]),
        ("CPU, 72 cores (1 node)", lj_cpu["hy"][72]),
        ("CPU, 720 cores (10 nodes)", lj_cpu["hy"][720]),
        ("GPU warp/row", gpu_lj["cuda_warp"][0]),
        ("GPU hybrid buckets", gpu_lj["cuda_hybrid"][0]),
        ("GPU thread/row", gpu_lj["cuda_thread"][0]),
        ("GPU cuSPARSE (reference)", gpu_lj["cuda_cusparse"][0])]
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw={"width_ratios": [1.25, 1]})
ax = axs[0]
ys = list(range(len(bars)))[::-1]
ax.barh(ys, [b[1] for b in bars], color=[C2] * 4 + [C1] * 4, height=0.62, edgecolor="white", linewidth=1)
ax.set_yticks(ys)
ax.set_yticklabels([b[0] for b in bars])
ax.set_xscale("log")
ax.set_xlabel("time per iteration (ms), soc-LiveJournal1")
ax.grid(axis="y", visible=False)
for y, b in zip(ys, bars):
    ax.text(b[1] * 1.08, y, f"{b[1]:.3g}", va="center", fontsize=6.5, color=INK2)
ax.set_xlim(1, 600)
ax.set_title("CPU vs one A40 GPU")


def mg(rs, graph, impl, nodes, ranks):
    sel = [r for r in rs if r["graph"] == graph and r["impl"] == impl and r["nodes"] == str(nodes)
           and r["ranks"] == str(ranks)]
    return med(sel, "t_iter", 1e3) if sel else None


ax = axs[1]
labels = ["1 GPU", "2 GPUs\none node", "2 GPUs\ntwo nodes"]
for i, (graph, c, short) in enumerate([("soc-LiveJournal1", C1, "LiveJournal"), ("com-friendster", C2, "Friendster")]):
    t = [mg(mg1, graph, "mpi_cuda_hybrid", 1, 1), mg(mg1, graph, "mpi_cuda_hybrid", 1, 2),
         mg(mg2, graph, "mpi_cuda_hybrid", 2, 2)]
    nums_key = "LJ" if i == 0 else "FR"
    num(f"mg{nums_key}one", t[0], "{:.2f}" if t[0] < 100 else "{:.0f}")
    num(f"mg{nums_key}two", t[1], "{:.2f}" if t[1] < 100 else "{:.0f}")
    num(f"mg{nums_key}xn", t[2], "{:.2f}" if t[2] < 100 else "{:.0f}")
    num(f"mg{nums_key}S", t[0] / t[1], "{:.2f}")
    num(f"mg{nums_key}Sx", t[0] / t[2], "{:.2f}")
    sp = [t[0] / x for x in t]
    xs = [j + (i - 0.5) * 0.34 for j in range(3)]
    ax.bar(xs, sp, width=0.32, color=c, edgecolor="white", linewidth=1, label=short)
    for x, v, tt in list(zip(xs, sp, t))[1:]:
        ax.text(x, v + 0.03, f"{v:.2f}x", ha="center", fontsize=6.5, color=INK2)
ax.axhline(2, color=GRAY, linestyle="--", linewidth=1)
ax.text(2.45, 2.02, "ideal 2x", fontsize=6.5, color=INK2, ha="right", va="bottom")
ax.set_xticks(range(3))
ax.set_xticklabels(labels)
ax.set_ylabel("speedup over 1 GPU")
ax.set_ylim(0, 2.6)
ax.grid(axis="x", visible=False)
ax.set_title("MPI+CUDA speedup")
ax.legend(loc="upper left", ncol=2, bbox_to_anchor=(0, 1.0))
save(fig, "gpu")
with open(f"{GEN}/tab_gpu.tex", "w") as f:
    nm = {"cuda_thread": "one thread per row", "cuda_warp": "one warp per row",
          "cuda_hybrid": "degree buckets (thread / warp / block)", "cuda_cusparse": "cuSPARSE SpMV (reference only)"}
    for impl in ["cuda_thread", "cuda_warp", "cuda_hybrid", "cuda_cusparse"]:
        t, bw = gpu_lj[impl]
        f.write(f"{nm[impl]} & {t:.2f} & {bw:.0f} & {bw/696*100:.0f}\\% \\\\\n")
for impl, k in [("cuda_thread", "Thr"), ("cuda_warp", "Warp"), ("cuda_hybrid", "Hyb"), ("cuda_cusparse", "Cusp")]:
    num(f"gpu{k}", gpu_lj[impl][0], "{:.2f}")
    num(f"gpu{k}bw", gpu_lj[impl][1], "{:.0f}")
num("gpuVsNode", lj_cpu["hy"][72] / gpu_lj["cuda_hybrid"][0], "{:.1f}")
num("gpuVsSocket", lj_cpu["hy"][36] / gpu_lj["cuda_hybrid"][0], "{:.1f}")
num("gpuVsSerial", real["soc-LiveJournal1"][0] / gpu_lj["cuda_hybrid"][0], "{:.0f}")
frmg = [r for r in mg1 if r["graph"] == "com-friendster" and r["ranks"] == "1" and r["impl"] == "mpi_cuda_hybrid"]
num("mgFRgbps", med(frmg, "gbps"), "{:.0f}")
frmg2 = [r for r in mg2 if r["graph"] == "com-friendster" and r["nodes"] == "2" and r["impl"] == "mpi_cuda_hybrid"]
frmg1n2 = [r for r in mg1 if r["graph"] == "com-friendster" and r["ranks"] == "2" and r["impl"] == "mpi_cuda_hybrid"]
num("mgFRcommone", per_iter(frmg1n2, "t_comm"), "{:.0f}")
num("mgFRcommtwo", per_iter(frmg2, "t_comm"), "{:.0f}")
num("mgFRstage", per_iter(frmg1n2, "t_stage"), "{:.0f}")
num("mgFRcomp", per_iter(frmg1n2, "t_comp"), "{:.0f}")
num("mgFRcompone", per_iter(frmg, "t_comp"), "{:.0f}")

# ---------------------------------------------------------------- design comparison (E3) and rank x thread sweep (E7)
des = rows("design_n4.csv")
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
ax = axs[0]
combos = [("vertex", "allgather"), ("vertex", "ghost"), ("edge", "allgather"), ("edge", "ghost")]
des_tab = {}
for i, (graph, c, short) in enumerate([("soc-LiveJournal1", C1, "LiveJournal"), ("com-friendster", C2, "Friendster")]):
    vals = []
    for part, comm in combos:
        sel = [r for r in des if r["graph"] == graph and r["part"] == part and r["comm"] == comm]
        vals.append(med(sel, "t_iter", 1e3))
        des_tab[(graph, part, comm)] = (med(sel, "t_iter", 1e3), per_iter(sel, "t_comp"), per_iter(sel, "t_comm"),
                                        med(sel, "imbalance"), med(sel, "comm_bytes_iter"))
    base = vals[0]
    xs = [j + (i - 0.5) * 0.34 for j in range(4)]
    ax.bar(xs, [v / base for v in vals], width=0.32, color=c, edgecolor="white", linewidth=1, label=short)
    for x, v in zip(xs, vals):
        ax.text(x, v / base + 0.03, f"{v:.0f}" if v >= 100 else f"{v:.1f}", ha="center", fontsize=6, color=INK2)
ax.set_xticks(range(4))
ax.set_xticklabels([f"{p}\n{c}" for p, c in combos])
ax.set_ylabel("time relative to vertex + allgather")
ax.grid(axis="x", visible=False)
ax.set_title("Partition $\\times$ exchange")
ax.set_ylim(0, 2.4)
ax.legend(loc="upper center", ncol=2)
ax = axs[1]
sw = rows("sweep_*_n4.csv")
for graph, c, mk, short in [("soc-LiveJournal1", C1, "o", "LiveJournal"), ("com-friendster", C2, "s", "Friendster")]:
    g = groups([r for r in sw if r["graph"] == graph], "ranks")
    pts = sorted((int(k[0]), med(v, "t_iter", 1e3), int(v[0]["threads"])) for k, v in g.items())
    t_best = min(p[1] for p in pts)
    ax.plot([p[0] for p in pts], [p[1] / t_best for p in pts], marker=mk, color=c, label=short)
ax.set_xscale("log", base=2)
xt = [4, 8, 16, 32, 72, 288]
ax.xaxis.set_major_locator(FixedLocator(xt))
ax.xaxis.set_minor_locator(NullLocator())
ax.set_xticklabels([f"{x}\n({288 // x})" for x in xt])
ax.set_xlabel("MPI ranks (threads per rank)")
ax.set_yscale("log")
ax.set_ylabel("time relative to best layout")
ax.set_title("Ranks $\\times$ threads")
ax.legend(loc="upper left")
plain_y(ax)
save(fig, "design")
with open(f"{GEN}/tab_design.tex", "w") as f:
    for graph in ["soc-LiveJournal1", "com-friendster"]:
        for part, comm in combos:
            t, cp, cm, imb, byt = des_tab[(graph, part, comm)]
            f.write(f"\\texttt{{{graph}}} & {part} & {comm} & {t:.1f} & {cp:.1f} & {cm:.1f} & {imb:.2f} & "
                    f"{byt/1e6:.0f} \\\\\n")
for graph, K in [("soc-LiveJournal1", "LJ"), ("com-friendster", "FR")]:
    num(f"dz{K}imbV", des_tab[(graph, "vertex", "ghost")][3], "{:.2f}")
    num(f"dz{K}imbE", des_tab[(graph, "edge", "ghost")][3], "{:.2f}")
    num(f"dz{K}byteA", des_tab[(graph, "edge", "allgather")][4] / 1e6, "{:.0f}")
    num(f"dz{K}byteG", des_tab[(graph, "edge", "ghost")][4] / 1e6, "{:.0f}")
    num(f"dz{K}compV", des_tab[(graph, "vertex", "ghost")][1], "{:.2f}")
    num(f"dz{K}compE", des_tab[(graph, "edge", "ghost")][1], "{:.2f}")

# ---------------------------------------------------------------- validation
val = {}
for f in ["val/val_cpu.csv", "val/val_gpu.csv"]:
    for r in csv.DictReader(open(f"{RES}/{f}")):
        val[(r["impl"], r["ranks"], r["part"], r["comm"])] = r
vs = {}
for f in glob.glob(f"{RES}/val/valscale*.csv") + glob.glob(f"{RES}/valscale/valscale*.csv"):
    for r in csv.DictReader(open(f)):
        vs[int(r["ranks"])] = r
cmp = {}
for f in ["val/compare_google.json", "val/compare_lj50.json", "valscale/validation_scale.json"]:
    p = f"{RES}/{f}"
    if os.path.exists(p):
        d = json.load(open(p))
        for k, v in d["vectors"].items():
            cmp[os.path.basename(k)] = v
with open(f"{GEN}/tab_val.tex", "w") as f:
    for b, lab in [("mpi_vertex_allgather.bin", "MPI+OpenMP, vertex, allgather"),
                   ("mpi_vertex_ghost.bin", "MPI+OpenMP, vertex, ghost"),
                   ("mpi_edge_allgather.bin", "MPI+OpenMP, edge, allgather"),
                   ("mpi_edge_ghost.bin", "MPI+OpenMP, edge, ghost"),
                   ("cuda_thread.bin", "CUDA, thread per row"), ("cuda_warp.bin", "CUDA, warp per row"),
                   ("cuda_hybrid.bin", "CUDA, degree buckets"), ("cuda_cusparse.bin", "CUDA, cuSPARSE"),
                   ("mpi_cuda_1gpu_hybrid.bin", "MPI+CUDA, 1 GPU"), ("mpi_cuda_2gpu_hybrid.bin", "MPI+CUDA, 2 GPUs")]:
        v = cmp.get(b)
        if v:
            f.write(f"web-Google & {lab} & {v['l1']:.1e} & {v['linf']:.1e} & "
                    f"{'yes' if v['top100_same_order'] else 'no'} \\\\\n")
    for p in sorted(x for x in vs if x > 1):
        v = cmp.get(f"mpi_lj50_p{p}.bin")
        if v:
            f.write(f"soc-LiveJournal1 & pure MPI, {p} ranks, {p // 72} nodes & {v['l1']:.1e} & {v['linf']:.1e} & "
                    f"{'yes' if v['top100_same_order'] else 'no'} \\\\\n")
num("valIters", int(val[("serial", "1", "none", "none")]["iters"]), "{:d}")
num("valScaleMax", max((p for p in vs if f"mpi_lj50_p{p}.bin" in cmp), default=0), "{:d}")
num("valMaxLone", max((v["l1"] for v in cmp.values()), default=0), "{:.1e}")

# failures that remain after the rerun (configurations without any CSV row) are listed in the report
# LaTeX command names cannot contain digits: a number such as stFRhy144 is stored under \csname and read in the
# text as \stFRhy{144} through one look-up macro per prefix.
with open(f"{GEN}/numbers.tex", "w") as f:
    prefixes = set()
    for k, v in sorted(nums.items()):
        base = k.rstrip("0123456789")
        if base != k:
            prefixes.add(base)
            f.write(f"\\expandafter\\def\\csname {k}\\endcsname{{{v}}}\n")
        else:
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    for b in sorted(prefixes):
        f.write(f"\\newcommand{{\\{b}}}[1]{{\\csname {b}#1\\endcsname}}\n")
print(f"wrote {len(os.listdir(FIG))} figures and {len(nums)} numbers")
missing = [n for n, v in nums.items() if v == "n/a"]
if missing:
    print("missing numbers:", missing)
