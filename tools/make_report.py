"""Build every figure and table of the report from the raw result files in results_final/.

Usage: python3 tools/make_report.py [results_final] [report]
Writes report/figures/*.pdf and report/generated/*.tex (table bodies and the \\newcommand numbers used in report.tex).
Every measured number in the report comes from here. A configuration is summarised by the median over its timed
repetitions; the warm-up repetition of the parallel programs is never written to the CSV files.
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

# Three categorical colours (checked for colour-blind separation) and gray for reference lines.
C1, C2, C3, GRAY, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 8.5, "axes.titlelocation": "left", "axes.titleweight": "bold",
    "axes.labelsize": 8.5, "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
    "grid.linewidth": 0.6, "lines.linewidth": 1.5, "lines.markersize": 4.5, "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight",
})
CORES = [1, 2, 4, 8, 16, 36, 72, 144, 288, 432, 576, 720]
MULTI = [72, 144, 288, 432, 576, 720]
TICKS = [1, 4, 16, 72, 144, 288, 720]
GNAME = {"lj": "soc-LiveJournal1", "fr": "com-friendster"}


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
    """median of a run total (t_comp, t_comm, t_wait, t_stage) divided by the iteration count, in ms"""
    v = [float(r[key]) / int(r["iters"]) * 1e3 for r in rs if r.get(key) not in (None, "")]
    return statistics.median(v) if v else None


def series(rs, exp, graph=None):
    """{cores: rows} for one experiment tag"""
    s = defaultdict(list)
    for r in rs:
        if r["exp"] == exp and (graph is None or r["graph"] == graph):
            s[int(r["cores"])].append(r)
    return dict(sorted(s.items()))


def logx_cores(ax, cores=TICKS, label="CPU cores"):
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(cores))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([str(c) for c in cores], rotation=0)
    ax.set_xlabel(label)


def plain_y(ax, ticks=(0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500)):
    """decimal tick labels on a log axis with a small range (avoids 6x10^0 style labels)"""
    ax.yaxis.set_major_locator(FixedLocator(list(ticks)))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())


def save(fig, name):
    # Multi-panel figures: label the panels (a), (b), ... so captions can refer to them.
    titled = [ax for ax in fig.axes if ax.get_title(loc="left")]
    if len(titled) > 1:
        for i, ax in enumerate(titled):
            ax.set_title(f"({'abcd'[i]}) {ax.get_title(loc='left')}", loc="left")
    fig.savefig(f"{FIG}/{name}.pdf")
    plt.close(fig)


nums = {}  # \newcommand macros for the text


def num(name, value, fmt="{:.2f}"):
    nums[name] = fmt.format(value) if value is not None else "n/a"


def ms(x):
    """a time in ms with three significant digits"""
    if x is None:
        return "--"
    return f"{x:.2f}" if x < 10 else (f"{x:.1f}" if x < 100 else f"{x:.0f}")


def eff(x):
    """efficiency: two decimals above 0.1, two significant digits below"""
    return f"{x:.2f}" if x >= 0.095 else f"{x:.2g}"


def spd(x):
    """speedup: one decimal below 10, integer above"""
    return f"{x:.1f}" if x < 9.95 else f"{x:.0f}"


def tex(name):
    return "\\texttt{" + name.replace("_", "\\_") + "}"


# ---------------------------------------------------------------- serial run time against problem size (S_*)
ser = rows("serial_size.csv")
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
spec = [("S_ef", "R-MAT, $N=2^{20}$, edge factor 4 to 64", C1, "o"),
        ("S_rmat", "R-MAT, $N$ and $M$ grow together", C2, "s"),
        ("S_real", "real graphs", C3, "^")]
serial_tab = []
for exp, label, c, mk in spec:
    g = groups([r for r in ser if r["exp"] == exp], "graph")
    pts = sorted((int(v[0]["M"]), int(v[0]["N"]), med(v, "t_iter", 1e3), k[0]) for k, v in g.items())
    M = [p[0] for p in pts]
    T = [p[2] for p in pts]
    nspe = [t * 1e6 / m for m, t in zip(M, T)]  # ns per edge
    ls = "-" if exp != "S_real" else "none"
    axs[0].plot(M, T, marker=mk, color=c, label=label, linestyle=ls)
    axs[1].plot(M, nspe, marker=mk, color=c, label=label, linestyle=ls)
    for m, n, t, name in pts:
        serial_tab.append((exp, name, n, m, t, t * 1e6 / m))
        if exp == "S_real":
            short = {"web-Google": "web-Google", "soc-LiveJournal1": "LiveJournal", "com-friendster": "Friendster"}[name]
            axs[1].annotate(short, (m, t * 1e6 / m), textcoords="offset points",
                            xytext=(-5, 0) if name == "com-friendster" else (5, -1),
                            ha="right" if name == "com-friendster" else "left", va="center", fontsize=6.5, color=INK2)
lo, hi = 1.5e6, 5e9
axs[0].plot([lo, hi], [x * 1e-6 * 2.0 for x in (lo, hi)], color=GRAY, linestyle="--", linewidth=1,
            label="slope 1 ($T \\propto M$)")
for ax in axs:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("edges $M$")
axs[0].set_ylabel("time per iteration (ms)")
axs[1].set_ylabel("time per edge (ns)")
axs[0].set_title("Time per iteration")
axs[1].set_title("Time per edge")
axs[0].legend(loc="upper left", fontsize=6.5)
axs[1].set_ylim(1.8, 9)
plain_y(axs[1], (2, 3, 4, 5, 6, 8))
save(fig, "serial_scaling")
with open(f"{GEN}/tab_serial.tex", "w") as f:
    names = {"S_ef": "$N$ fixed", "S_rmat": "$N$, $M$ grow", "S_real": "real"}
    last = None
    for exp, name, n, m, t, ns in serial_tab:
        if last and exp != last:
            f.write("\\addlinespace[2pt]\n")
        last = exp
        f.write(f"{names[exp]} & {tex(name)} & {n/1e6:.2f} & {m/1e6:.1f} & {m/n:.1f} & {n*8/1e6:.0f} & "
                f"{ms(t)} & {ns:.2f} \\\\\n")
sv = {name: (n, m, t, ns) for exp, name, n, m, t, ns in serial_tab}
real = {name: (t, ns) for exp, name, n, m, t, ns in serial_tab if exp == "S_real"}
num("serLJms", real["soc-LiveJournal1"][0], "{:.0f}")
num("serLJns", real["soc-LiveJournal1"][1], "{:.1f}")
num("serFRs", real["com-friendster"][0] / 1e3, "{:.1f}")
num("serFRns", real["com-friendster"][1], "{:.1f}")
ef = sorted((m, n, t, ns) for exp, name, n, m, t, ns in serial_tab if exp == "S_ef")
rm = sorted((m, n, t, ns) for exp, name, n, m, t, ns in serial_tab if exp == "S_rmat")
num("serEFmin", min(p[3] for p in ef), "{:.1f}")
num("serEFmax", max(p[3] for p in ef), "{:.1f}")
num("serEFtlo", ef[0][2], "{:.0f}")
num("serEFthi", ef[-1][2], "{:.0f}")
num("serEFmratio", ef[-1][0] / ef[0][0], "{:.0f}")
num("serEFtratio", ef[-1][2] / ef[0][2], "{:.0f}")
# least-squares line T = a M + b through the five fixed-N graphs: a in ns per edge, b / N in ns per vertex
mx, my = statistics.mean(p[0] for p in ef), statistics.mean(p[2] for p in ef)
slope = sum((p[0] - mx) * (p[2] - my) for p in ef) / sum((p[0] - mx) ** 2 for p in ef)
num("serEFslope", slope * 1e6, "{:.1f}")
num("serEFicept", (my - slope * mx) * 1e6 / ef[0][1], "{:.0f}")
num("serRMlo", rm[0][3], "{:.1f}")
num("serRMhi", rm[-1][3], "{:.1f}")
num("serRMtlo", rm[0][2], "{:.1f}")
num("serRMthi", rm[-1][2] / 1e3, "{:.1f}")
num("serRMmratio", rm[-1][0] / rm[0][0], "{:.0f}")
num("serRMtratio", rm[-1][2] / rm[0][2], "{:.0f}")
num("serPone", sv["rmat_p1"][2], "{:.1f}")

# ---------------------------------------------------------------- strong scaling (E1)
strong = {"lj": rows("strong_lj_n*.csv"), "fr": rows("strong_fr_n*.csv")}
title = {"lj": "soc-LiveJournal1 (68M edges)", "fr": "com-friendster (3.6G edges)"}
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7))
st = {}
for ax, key in zip(axs, ["lj", "fr"]):
    hy = series(strong[key], "E1_hybrid")
    mp = series(strong[key], "E1_mpi")
    mp = {1: hy[1], **mp}  # the one-core run is the same program in both layouts
    t1 = med(hy[1], "t_iter", 1e3)
    for s, lab, c, mk in [(hy, "MPI+OpenMP (hybrid layout)", C1, "o"), (mp, "pure MPI (one rank per core)", C2, "s")]:
        ax.plot(sorted(s), [med(s[c_], "t_iter", 1e3) for c_ in sorted(s)], marker=mk, color=c, label=lab)
    ax.plot([1, 720], [t1, t1 / 720], color=GRAY, linestyle="--", linewidth=1, label="ideal ($T_1/p$)")
    ax.axvline(72, color=GRAY, linewidth=0.8, linestyle=":")
    ax.text(66, 0.97, "one node (72 cores)", transform=ax.get_xaxis_transform(), rotation=90, fontsize=6.5,
            color=INK2, ha="right", va="top")
    logx_cores(ax)
    ax.set_yscale("log")
    ax.set_ylabel("time per iteration (ms)")
    ax.set_title(title[key])
    st[key] = {"t1": t1, "hy": {c_: med(v, "t_iter", 1e3) for c_, v in hy.items()},
               "mp": {c_: med(v, "t_iter", 1e3) for c_, v in mp.items()}, "hyr": hy, "mpr": mp}
fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.11))
save(fig, "strong_time")

fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw={"wspace": 0.27})
for key, ls in [("lj", "-"), ("fr", "--")]:
    t1 = st[key]["t1"]
    for mode, c, mk, lab in [("hy", C1, "o", "MPI+OpenMP"), ("mp", C2, "s", "pure MPI")]:
        d = st[key][mode]
        xs = sorted(d)
        sp = [t1 / d[x] for x in xs]
        face = c if key == "lj" else "white"
        axs[0].plot(xs, sp, marker=mk, color=c, linestyle=ls, label=f"{lab}, {GNAME[key]}", markerfacecolor=face)
        axs[1].plot(xs, [s_ / x for s_, x in zip(sp, xs)], marker=mk, color=c, linestyle=ls, markerfacecolor=face)
axs[0].plot([1, 720], [1, 720], color=GRAY, linestyle=":", linewidth=1, label="ideal")
for ax in axs:
    logx_cores(ax)
    ax.set_yscale("log")
axs[0].set_ylabel("speedup $S_p = T_1 / T_p$")
axs[1].set_ylabel("efficiency $E_p = S_p / p$")
axs[0].set_title("Speedup")
axs[1].set_title("Parallel efficiency")
axs[0].legend(loc="upper left", fontsize=6.5)
save(fig, "strong_speedup")

with open(f"{GEN}/tab_strong.tex", "w") as f:
    for c_ in CORES:
        cells = [str(c_)]
        for key in ["lj", "fr"]:
            t1 = st[key]["t1"]
            for mode in ["hy", "mp"]:
                t = st[key][mode][c_]
                cells += [ms(t), spd(t1 / t), eff(t1 / t / c_)]
        if c_ == 144:
            f.write("\\addlinespace[2pt]\n")
        f.write(" & ".join(cells) + " \\\\\n")
for key in ["lj", "fr"]:
    K = key.upper()
    hy, mp, t1 = st[key]["hy"], st[key]["mp"], st[key]["t1"]
    best, bestmp = min(hy, key=lambda c_: hy[c_]), min(mp, key=lambda c_: mp[c_])
    num(f"st{K}one", t1, "{:.0f}")
    num(f"st{K}best", hy[best], "{:.2f}" if hy[best] < 100 else "{:.0f}")
    num(f"st{K}bestc", best, "{:d}")
    num(f"st{K}bestS", t1 / hy[best], "{:.0f}")
    num(f"st{K}bestE", t1 / hy[best] / best, "{:.2f}")
    num(f"st{K}mpbest", mp[bestmp], "{:.2f}" if mp[bestmp] < 100 else "{:.0f}")
    num(f"st{K}mpbestc", bestmp, "{:d}")
    num(f"st{K}mpbestS", t1 / mp[bestmp], "{:.0f}")
    num(f"st{K}nodeS", hy[72] / hy[best], "{:.1f}")
    multi = [hy[c_] for c_ in MULTI]
    num(f"st{K}multimin", min(multi), "{:.1f}" if min(multi) < 100 else "{:.0f}")
    num(f"st{K}multimax", max(multi), "{:.1f}" if max(multi) < 100 else "{:.0f}")
    for c_ in CORES:
        num(f"st{K}hy{c_}", hy[c_], "{:.2f}" if hy[c_] < 100 else "{:.0f}")
        num(f"st{K}hyS{c_}", t1 / hy[c_], "{:.1f}" if t1 / hy[c_] < 100 else "{:.0f}")
        num(f"st{K}hyE{c_}", t1 / hy[c_] / c_, "{:.2f}")
        num(f"st{K}mp{c_}", mp[c_], "{:.2f}" if mp[c_] < 100 else "{:.0f}")
        num(f"st{K}mpS{c_}", t1 / mp[c_], "{:.1f}")
        num(f"st{K}mpE{c_}", t1 / mp[c_] / c_, "{:.2f}" if t1 / mp[c_] / c_ >= 0.095 else "{:.2g}")
    effs = [t1 / hy[c_] / c_ for c_ in (2, 4, 8, 16)]
    num(f"st{K}effLo", min(effs), "{:.2f}")
    num(f"st{K}effHi", max(effs), "{:.2f}")
    num(f"st{K}ratio", max(mp[c_] / hy[c_] for c_ in MULTI), "{:.0f}")
    num(f"st{K}ratioc", max(MULTI, key=lambda c_: mp[c_] / hy[c_]), "{:d}")
    num(f"base{K}", (t1 / real[GNAME[key]][0] - 1) * 100, "{:.0f}")
num("stFRones", st["fr"]["t1"] / 1e3, "{:.1f}")
num("stLJmpslow", max(st["lj"]["mp"][c_] for c_ in (432, 576, 720)), "{:.0f}")
num("stLJmpfast", min(st["lj"]["mp"][c_] for c_ in (432, 576, 720)), "{:.0f}")

# ---------------------------------------------------------------- computation and communication time (E1, E8)
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7))
cm = {}
for ax, key in zip(axs, ["lj", "fr"]):
    cm[key] = {}
    for mode, c, mk, lab in [("hyr", C1, "o", "hybrid"), ("mpr", C2, "s", "pure MPI")]:
        s = st[key][mode]
        comp = {x: per_iter(s[x], "t_comp") for x in sorted(s)}
        comm = {x: per_iter(s[x], "t_comm") for x in sorted(s) if int(s[x][0]["ranks"]) > 1}
        vol = {x: med(s[x], "comm_bytes_iter") for x in comm}
        ranks = {x: int(s[x][0]["ranks"]) for x in comm}
        cm[key][mode[:2]] = {"comp": comp, "comm": comm, "vol": vol, "ranks": ranks}
        xs = [x for x in sorted(comp) if x > 1 or mode == "hyr"]
        ax.plot(xs, [comp[x] for x in xs], marker=mk, color=c, label=f"{lab}, computation")
        ax.plot(sorted(comm), [comm[x] for x in sorted(comm)], marker=mk, color=c, linestyle="--",
                markerfacecolor="white", label=f"{lab}, communication")
    logx_cores(ax)
    ax.set_yscale("log")
    ax.set_ylabel("ms per iteration")
    ax.set_title(GNAME[key])
fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, -0.13),
           columnspacing=1.6, handlelength=2.4)
save(fig, "comm_time")

with open(f"{GEN}/tab_comm.tex", "w") as f:
    for c_ in MULTI:
        cells = [str(c_)]
        for key in ["lj", "fr"]:
            for mode in ["hy", "mp"]:
                d = cm[key][mode]
                cells += [ms(d["comp"][c_]), ms(d["comm"][c_]), f"{d['vol'][c_] / 1e6:.0f}"]
        f.write(" & ".join(cells) + " \\\\\n")
for key in ["lj", "fr"]:
    K = key.upper()
    hy, mp = cm[key]["hy"], cm[key]["mp"]
    for c_ in MULTI:
        num(f"cm{K}hyComp{c_}", hy["comp"][c_], "{:.2f}" if hy["comp"][c_] < 10 else "{:.0f}")
        num(f"cm{K}hyComm{c_}", hy["comm"][c_], "{:.2f}" if hy["comm"][c_] < 10 else "{:.0f}")
        num(f"cm{K}mpComp{c_}", mp["comp"][c_], "{:.2f}" if mp["comp"][c_] < 10 else "{:.0f}")
        num(f"cm{K}mpComm{c_}", mp["comm"][c_], "{:.2f}" if mp["comm"][c_] < 10 else "{:.0f}")
        num(f"cm{K}hyVol{c_}", hy["vol"][c_] / 1e6, "{:.0f}")
        num(f"cm{K}mpVol{c_}", mp["vol"][c_] / 1e6, "{:.0f}")
        # ghost data received by one rank per iteration (MB) and the rate at which the exchange moves it (GB/s)
        num(f"cm{K}hyPer{c_}", hy["vol"][c_] / hy["ranks"][c_] / 1e6, "{:.0f}" if key == "fr" else "{:.1f}")
        num(f"cm{K}hyRate{c_}", hy["vol"][c_] / hy["ranks"][c_] / hy["comm"][c_] / 1e6, "{:.1f}")
    rates = [hy["vol"][c_] / hy["ranks"][c_] / hy["comm"][c_] / 1e6 for c_ in MULTI]
    num(f"cm{K}rateMin", min(rates), "{:.1f}")
    num(f"cm{K}rateMax", max(rates), "{:.1f}")
    ratio = [mp["vol"][c_] / hy["vol"][c_] for c_ in MULTI]
    num(f"cm{K}volMin", min(ratio), "{:.0f}")
    num(f"cm{K}volMax", max(ratio), "{:.0f}")
    num(f"cm{K}hyVolGB", hy["vol"][720] / 1e9, "{:.1f}")
    num(f"cm{K}mpVolGB", mp["vol"][720] / 1e9, "{:.0f}" if key == "fr" else "{:.2f}")
    num(f"cm{K}mpRate", mp["vol"][720] / 720 / mp["comm"][720] / 1e3, "{:.0f}")  # MB/s per rank at 720 ranks
    num(f"cm{K}mpPer", mp["vol"][720] / 720 / 1e6, "{:.1f}" if key == "lj" else "{:.0f}")
    n_vert = int(st[key]["hyr"][720][0]["N"])
    num(f"cm{K}ghostFrac", hy["vol"][720] / 20 / 8 / n_vert * 100, "{:.0f}")
    num(f"cm{K}edgesPerGhost", int(st[key]["hyr"][144][0]["M"]) / (hy["vol"][144] / 8), "{:.0f}")
num("cmALLrateMin", min(float(nums[f"cm{K}rateMin"]) for K in ("LJ", "FR")), "{:.1f}")
num("cmALLrateMax", max(float(nums[f"cm{K}rateMax"]) for K in ("LJ", "FR")), "{:.1f}")

# --split runs: a barrier in front of every MPI call separates the wait for the slowest rank from the data movement
split = {"lj": rows("split_lj_n*.csv"), "fr": rows("split_fr_n*.csv")}
for key in ["lj", "fr"]:
    K = key.upper()
    s = series(split[key], "E8_hybrid")
    for c_ in sorted(s):
        for part in ["comp", "wait", "comm"]:
            v = per_iter(s[c_], f"t_{part}")
            num(f"sp{K}{part}{c_}", v, "{:.2f}" if v < 10 else "{:.0f}")
    waits = [per_iter(s[c_], "t_wait") for c_ in sorted(s)]
    num(f"sp{K}waitMin", min(waits), "{:.1f}" if min(waits) < 10 else "{:.0f}")
    num(f"sp{K}waitMax", max(waits), "{:.1f}" if max(waits) < 10 else "{:.0f}")

# The same pure-MPI configuration (soc-LiveJournal1, 288 ranks, edge partition, ghost exchange) was launched five
# times in different experiments; the spread of its medians shows the run-to-run variation over TCP.
lj288 = []
for pat, exp in [("strong_lj_n4.csv", "E1_mpi"), ("split_lj_n4.csv", "E8_mpi"), ("sweep_lj_n4.csv", "E7"),
                 ("design_n4.csv", "E3"), ("val/valscale_n4.csv", "valscale")]:
    sel = [r for r in rows(pat) if r["exp"] == exp and r["graph"] == "soc-LiveJournal1" and r["ranks"] == "288"
           and r["part"] == "edge" and r["comm"] == "ghost"]
    if sel:
        lj288.append(med(sel, "t_iter", 1e3))
num("mpLJvarMin", min(lj288), "{:.0f}")
num("mpLJvarMax", max(lj288), "{:.0f}")
num("mpLJvarN", len(lj288), "{:d}")

# ---------------------------------------------------------------- weak scaling (E2)
weak = rows("weak_n*.csv")
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"wspace": 0.27})
wk = {}
for mode, c, mk, lab in [("E2_hybrid", C1, "o", "MPI+OpenMP (hybrid layout)"), ("E2_mpi", C2, "s", "pure MPI (one rank per core)")]:
    s = series(weak, mode)
    if mode == "E2_mpi":
        s = {1: series(weak, "E2_hybrid")[1], **s}
    xs = sorted(s)
    ts = [med(s[x], "t_iter", 1e3) for x in xs]
    wk[mode] = {"t": dict(zip(xs, ts)), "rows": s}
    axs[0].plot(xs, ts, marker=mk, color=c, label=lab)
    axs[1].plot(xs, [ts[0] / t for t in ts], marker=mk, color=c, label=lab)
t1w = wk["E2_hybrid"]["t"][1]
axs[0].axhline(t1w, color=GRAY, linestyle="--", linewidth=1, label="ideal (constant time)")
axs[1].axhline(1, color=GRAY, linestyle="--", linewidth=1)
for ax in axs:
    logx_cores(ax)
    ax.set_yscale("log")
axs[0].set_ylabel("time per iteration (ms)")
axs[1].set_ylabel("weak-scaling efficiency $T_1 / T_p$")
axs[0].set_title("Time per iteration")
axs[1].set_title("Weak-scaling efficiency")
axs[0].legend(loc="upper left")
save(fig, "weak")
epc = []
with open(f"{GEN}/tab_weak.tex", "w") as f:
    for c_ in CORES:
        hr, mr = wk["E2_hybrid"]["rows"][c_], wk["E2_mpi"]["rows"][c_]
        h, m = wk["E2_hybrid"]["t"][c_], wk["E2_mpi"]["t"][c_]
        n_, m_ = int(hr[0]["N"]), int(hr[0]["M"])
        epc.append(m_ / c_ / 1e6)
        hcomm = per_iter(hr, "t_comm") if int(hr[0]["ranks"]) > 1 else None
        mcomm = per_iter(mr, "t_comm") if c_ > 1 else None
        if c_ == 144:
            f.write("\\addlinespace[2pt]\n")
        f.write(f"{c_} & {n_/1e6:.2f} & {m_/1e6:.0f} & {m_/c_/1e6:.2f} & {ms(h)} & {ms(per_iter(hr, 't_comp'))} & "
                f"{ms(hcomm)} & {eff(t1w / h)} & {ms(m)} & {ms(mcomm)} & {eff(t1w / m)} \\\\\n")
num("wkOne", t1w, "{:.2f}")
num("wkEpcMin", min(epc), "{:.2f}")
num("wkEpcMax", max(epc), "{:.2f}")
for c_ in CORES[1:]:
    hr = wk["E2_hybrid"]["rows"][c_]
    num(f"wkHy{c_}", wk["E2_hybrid"]["t"][c_], "{:.1f}")
    num(f"wkHyE{c_}", t1w / wk["E2_hybrid"]["t"][c_], "{:.2f}" if t1w / wk["E2_hybrid"]["t"][c_] >= 0.095 else "{:.2g}")
    num(f"wkHyComp{c_}", per_iter(hr, "t_comp"), "{:.1f}")
    num(f"wkHyComm{c_}", per_iter(hr, "t_comm"), "{:.1f}")
    num(f"wkMp{c_}", wk["E2_mpi"]["t"][c_], "{:.1f}" if wk["E2_mpi"]["t"][c_] < 100 else "{:.0f}")
num("wkMpE720", t1w / wk["E2_mpi"]["t"][720], "{:.4f}")
num("wkSerS720", sv["rmat_p720"][2] / wk["E2_hybrid"]["t"][720], "{:.0f}")
num("wkSerS288", sv["rmat_p288"][2] / wk["E2_hybrid"]["t"][288], "{:.0f}")
num("wkSerT720", sv["rmat_p720"][2] / 1e3, "{:.1f}")

# ---------------------------------------------------------------- OpenMP: memory bandwidth and ranks x threads (E7)
stream = list(csv.DictReader(open(f"{RES}/stream_cpu.csv")))
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.45), gridspec_kw={"wspace": 0.42})
ax = axs[0]
th = [int(r["threads"]) for r in stream]
ax.plot(th, [float(r["triad_gbs"]) for r in stream], marker="^", color=GRAY, label="STREAM triad")
for key, c, mk, short in [("lj", C1, "o", "PageRank, LiveJournal"), ("fr", C2, "s", "PageRank, Friendster")]:
    s = st[key]["hyr"]
    xs = [x for x in sorted(s) if x <= 72]
    ax.plot(xs, [med(s[x], "gbps") for x in xs], marker=mk, color=c, label=short)
logx_cores(ax, [1, 4, 16, 72], "cores on one node")
ax.set_yscale("log")
ax.set_ylabel("GB/s")
ax.set_title("Memory bandwidth")
ax.legend(loc="upper left", fontsize=6.3, handlelength=1.4, borderaxespad=0.2)
ax.set_ylim(1, 3000)
sw = {}
for ax, nd, cores_, xt in [(axs[1], "n1", 72, [1, 2, 4, 8, 18, 72]), (axs[2], "n4", 288, [4, 8, 16, 32, 72, 288])]:
    for key, c, mk, short in [("lj", C1, "o", "LiveJournal"), ("fr", C2, "s", "Friendster")]:
        g = groups(rows(f"sweep_{key}_{nd}.csv"), "ranks")
        pts = sorted((int(k[0]), med(v, "t_iter", 1e3)) for k, v in g.items())
        sw[(key, nd)] = dict(pts)
        best = min(p[1] for p in pts)
        ax.plot([p[0] for p in pts], [p[1] / best for p in pts], marker=mk, color=c, label=short)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(xt))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([str(x) for x in xt])
    ax.set_xlabel("MPI ranks")
    ax.set_yscale("log")
    ax.set_ylabel("time relative to best")
    ax.set_title(f"{cores_} cores ({'one node' if cores_ == 72 else 'four nodes'})")
    plain_y(ax, (1, 1.5, 2, 3, 5, 10))
axs[1].set_ylim(0.93, 2.2)
axs[2].set_ylim(0.9, 16)
axs[1].legend(loc="lower right", fontsize=6.5)
save(fig, "openmp")
num("streamOne", float(stream[0]["triad_gbs"]), "{:.0f}")
num("streamFull", float(stream[-1]["triad_gbs"]), "{:.0f}")
num("streamHalf", float([r for r in stream if r["threads"] == "36"][0]["triad_gbs"]), "{:.0f}")
num("streamPerCore", float(stream[-1]["triad_gbs"]) / 72, "{:.1f}")
num("streamDrop", float(stream[0]["triad_gbs"]) * 72 / float(stream[-1]["triad_gbs"]), "{:.1f}")
num("prLJbwOne", med(st["lj"]["hyr"][1], "gbps"), "{:.1f}")
num("prFRbwOne", med(st["fr"]["hyr"][1], "gbps"), "{:.1f}")
num("prLJbw36", med(st["lj"]["hyr"][36], "gbps"), "{:.0f}")
num("prFRbw36", med(st["fr"]["hyr"][36], "gbps"), "{:.0f}")
num("prFRbw72", med(st["fr"]["hyr"][72], "gbps"), "{:.0f}")
num("prFRshare", med(st["fr"]["hyr"][72], "gbps") / float(stream[-1]["triad_gbs"]) * 100, "{:.0f}")
num("prLJshare", med(st["lj"]["hyr"][36], "gbps") / float([r for r in stream if r["threads"] == "36"][0]["triad_gbs"]) * 100, "{:.0f}")
for key in ["lj", "fr"]:
    K = key.upper()
    for r_, t in sw[(key, "n1")].items():
        num(f"sw{K}one{r_}", t, "{:.2f}" if t < 100 else "{:.0f}")
    for r_, t in sw[(key, "n4")].items():
        num(f"sw{K}four{r_}", t, "{:.2f}" if t < 100 else "{:.0f}")
    best = min(sw[(key, "n4")].values())
    num(f"sw{K}fourWorst", max(sw[(key, "n4")].values()) / best, "{:.0f}" if key == "lj" else "{:.1f}")
num("swFRnuma", sw[("fr", "n1")][1] / sw[("fr", "n1")][2], "{:.1f}")
num("swLJnuma", sw[("lj", "n1")][2] / sw[("lj", "n1")][1], "{:.1f}")

# ---------------------------------------------------------------- partition x exchange on 288 ranks (E3)
des = rows("design_n4.csv")
combos = [("vertex", "allgather"), ("vertex", "ghost"), ("edge", "allgather"), ("edge", "ghost")]
dz = {}
with open(f"{GEN}/tab_design.tex", "w") as f:
    for graph in ["soc-LiveJournal1", "com-friendster"]:
        if graph == "com-friendster":
            f.write("\\addlinespace[2pt]\n")
        for part, comm in combos:
            sel = [r for r in des if r["graph"] == graph and r["part"] == part and r["comm"] == comm]
            t, cp, cc = med(sel, "t_iter", 1e3), per_iter(sel, "t_comp"), per_iter(sel, "t_comm")
            imb, byt = med(sel, "imbalance"), med(sel, "comm_bytes_iter")
            dz[(graph, part, comm)] = (t, cp, cc, imb, byt)
            f.write(f"{tex(graph)} & {part} & {comm} & {ms(t)} & {ms(cp)} & {ms(cc)} & {imb:.2f} & {byt/1e6:,.0f} \\\\\n"
                    .replace(",", "{,}"))
for graph, K in [("soc-LiveJournal1", "LJ"), ("com-friendster", "FR")]:
    num(f"dz{K}imbV", dz[(graph, "vertex", "ghost")][3], "{:.2f}")
    num(f"dz{K}imbE", dz[(graph, "edge", "ghost")][3], "{:.2f}")
    num(f"dz{K}compV", dz[(graph, "vertex", "ghost")][1], "{:.0f}" if K == "FR" else "{:.1f}")
    num(f"dz{K}compE", dz[(graph, "edge", "ghost")][1], "{:.0f}" if K == "FR" else "{:.1f}")
    num(f"dz{K}tEA", dz[(graph, "edge", "allgather")][0], "{:.0f}")
    num(f"dz{K}tEG", dz[(graph, "edge", "ghost")][0], "{:.0f}")
    num(f"dz{K}tVG", dz[(graph, "vertex", "ghost")][0], "{:.0f}")
    num(f"dz{K}gbA", dz[(graph, "edge", "allgather")][4] / 1e9, "{:.0f}")
    num(f"dz{K}mbG", dz[(graph, "edge", "ghost")][4] / 1e6, "{:.0f}")
    num(f"dz{K}volRatio", dz[(graph, "edge", "allgather")][4] / dz[(graph, "edge", "ghost")][4], "{:.0f}")

# ---------------------------------------------------------------- GPU (E4, E6) and the GPU memory probe
gbw = list(csv.DictReader(open(f"{RES}/gpu_bw.csv")))[0]
num("gpuCopy", float(gbw["copy_gbs"]), "{:.0f}")
num("gpuTriad", float(gbw["triad_gbs"]), "{:.0f}")
num("gpuGather", float(gbw["random_gather_gbs"]), "{:.0f}")
num("gpuGatherRatio", float(gbw["triad_gbs"]) / float(gbw["random_gather_gbs"]), "{:.1f}")
num("gpuVsCpuTriad", float(gbw["triad_gbs"]) / float(stream[-1]["triad_gbs"]), "{:.1f}")
kern = groups(rows("e4.csv"), "impl")
mg1 = rows("multigpu_1node.csv")
mg2 = rows("multigpu_2node.csv")


def mg(rs, graph, impl, nodes, ranks):
    return [r for r in rs if r["graph"] == graph and r["impl"] == impl and r["nodes"] == str(nodes)
            and r["ranks"] == str(ranks)]


LJ, FR = "soc-LiveJournal1", "com-friendster"
gpu_lj = {impl: (med(kern[(impl,)], "t_iter", 1e3), med(kern[(impl,)], "gbps"))
          for impl in ["cuda_thread", "cuda_warp", "cuda_hybrid", "cuda_cusparse"]}
for impl in ["mpi_cuda_thread", "mpi_cuda_hybrid"]:
    sel = mg(mg1, LJ, impl, 1, 1)
    gpu_lj[impl] = (med(sel, "t_iter", 1e3), med(sel, "gbps"))
lj72 = sw[("lj", "n1")][1]  # one rank with 72 threads, the fastest one-node layout for this graph
bars = [("CPU: serial, 1 core", real[LJ][0], C2),
        ("CPU: 36 cores", st["lj"]["hy"][36], C2),
        ("CPU: 72 cores", lj72, C2),
        ("CUDA: warp per row", gpu_lj["cuda_warp"][0], C1),
        ("CUDA: degree buckets", gpu_lj["cuda_hybrid"][0], C1),
        ("CUDA: thread per row", gpu_lj["cuda_thread"][0], C1),
        ("CUDA: cuSPARSE", gpu_lj["cuda_cusparse"][0], GRAY),
        ("MPI+CUDA: thread per row", gpu_lj["mpi_cuda_thread"][0], C1),
        ("MPI+CUDA: degree buckets", gpu_lj["mpi_cuda_hybrid"][0], C1)]
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={"width_ratios": [1.2, 1]})
ax = axs[0]
ys = list(range(len(bars)))[::-1]
ax.barh(ys, [b[1] for b in bars], color=[b[2] for b in bars], height=0.62, edgecolor="white", linewidth=1)
ax.set_yticks(ys)
ax.set_yticklabels([b[0] for b in bars])
ax.set_xscale("log")
ax.set_xlabel("time per iteration (ms)")
ax.grid(axis="y", visible=False)
for y, b in zip(ys, bars):
    ax.text(b[1] * 1.08, y, f"{b[1]:.3g}", va="center", fontsize=6.5, color=INK2)
ax.set_xlim(1, 600)
ax.set_title("soc-LiveJournal1: CPU and one A40")
ax = axs[1]
labels = ["1 GPU", "2 GPUs\none node", "2 GPUs\ntwo nodes"]
for i, (graph, c, short, K) in enumerate([(LJ, C1, "LiveJournal", "LJ"), (FR, C2, "Friendster", "FR")]):
    sel = [mg(mg1, graph, "mpi_cuda_hybrid", 1, 1), mg(mg1, graph, "mpi_cuda_hybrid", 1, 2),
           mg(mg2, graph, "mpi_cuda_hybrid", 2, 2)]
    t = [med(s_, "t_iter", 1e3) for s_ in sel]
    for tag, s_, tv in zip(["one", "two", "xn"], sel, t):
        num(f"mg{K}{tag}", tv, "{:.2f}" if tv < 100 else "{:.0f}")
        for part in ["comp", "comm", "stage"]:
            v = per_iter(s_, f"t_{part}")
            num(f"mg{K}{tag}{part}", v, "{:.2f}" if v < 10 else "{:.0f}")
    num(f"mg{K}S", t[0] / t[1], "{:.2f}")
    num(f"mg{K}Sx", t[0] / t[2], "{:.2f}")
    num(f"mg{K}E", t[0] / t[1] / 2, "{:.2f}")
    num(f"mg{K}vol", med(sel[1], "comm_bytes_iter") / 1e6, "{:.0f}")
    sp = [t[0] / x for x in t]
    xs = [j + (i - 0.5) * 0.34 for j in range(3)]
    ax.bar(xs, sp, width=0.32, color=c, edgecolor="white", linewidth=1, label=short)
    for x, v in list(zip(xs, sp))[1:]:
        ax.text(x, v + 0.03, f"{v:.2f}", ha="center", fontsize=6.5, color=INK2)
ax.axhline(2, color=GRAY, linestyle="--", linewidth=1)
ax.text(2.45, 2.03, "ideal for 2 GPUs", fontsize=6.5, color=INK2, ha="right", va="bottom")
ax.set_xticks(range(3))
ax.set_xticklabels(labels)
ax.set_ylabel("speedup over one GPU")
ax.set_ylim(0, 2.6)
ax.grid(axis="x", visible=False)
ax.set_title("MPI+CUDA on two GPUs")
ax.legend(loc="upper left", ncol=2, bbox_to_anchor=(0, 1.0))
save(fig, "gpu")
with open(f"{GEN}/tab_gpu.tex", "w") as f:
    nm = [("cuda_thread", "CUDA", "one thread per row"), ("cuda_warp", "CUDA", "one warp per row"),
          ("cuda_hybrid", "CUDA", "degree buckets"), ("cuda_cusparse", "CUDA", "\\texttt{cusparseSpMV} (reference)"),
          ("mpi_cuda_thread", "MPI+CUDA, 1 GPU", "one thread per row"),
          ("mpi_cuda_hybrid", "MPI+CUDA, 1 GPU", "degree buckets")]
    for impl, prog, k in nm:
        t, bw = gpu_lj[impl]
        if impl == "mpi_cuda_thread":
            f.write("\\addlinespace[2pt]\n")
        f.write(f"{prog} & {k} & {t:.2f} & {bw:.0f} \\\\\n")
for impl, k in [("cuda_thread", "Thr"), ("cuda_warp", "Warp"), ("cuda_hybrid", "Hyb"), ("cuda_cusparse", "Cusp"),
                ("mpi_cuda_thread", "MThr"), ("mpi_cuda_hybrid", "MHyb")]:
    num(f"gpu{k}", gpu_lj[impl][0], "{:.2f}")
    num(f"gpu{k}bw", gpu_lj[impl][1], "{:.0f}")
best_gpu = gpu_lj["mpi_cuda_hybrid"][0]
num("gpuWarpSlow", gpu_lj["cuda_warp"][0] / gpu_lj["cuda_thread"][0], "{:.0f}")
num("gpuVsSerial", real[LJ][0] / best_gpu, "{:.0f}")
num("gpuVsSocket", st["lj"]["hy"][36] / best_gpu, "{:.1f}")
num("gpuVsNode", lj72 / best_gpu, "{:.2f}")
num("gpuFRvsNode", med(mg(mg1, FR, "mpi_cuda_hybrid", 1, 1), "t_iter", 1e3) / st["fr"]["hy"][72], "{:.1f}")
num("gpuFRbw", med(mg(mg1, FR, "mpi_cuda_hybrid", 1, 1), "gbps"), "{:.0f}")

# ---------------------------------------------------------------- web-Google: convergence and rank against in-degree
gg = json.load(open(f"{RES}/val/google_analysis.json"))
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
ax = axs[0]
it = [c["iter"] for c in gg["convergence"]]
rs_ = [c["resid"] for c in gg["convergence"]]
ax.plot(it, rs_, color=C1, label="measured residual")
ax.plot(it, [rs_[19] * 0.85 ** (i - 20) for i in it], color=GRAY, linestyle="--", linewidth=1,
        label="slope of $0.85^k$")
ax.set_yscale("log")
ax.set_xlabel("iteration $k$")
ax.set_ylabel("residual $\\|r^{(k)}-r^{(k-1)}\\|_1$")
ax.set_title("Convergence")
ax.legend(loc="upper right")
ax = axs[1]
bins = gg["pr_by_indeg"]
lab = [str(b["lo"]) if b["lo"] == b["hi"] else f"{b['lo']}" for b in bins]
ax.plot(range(len(bins)), [b["mean_pr"] * gg["N"] for b in bins], marker="o", color=C1)
ax.axhline(1, color=GRAY, linestyle="--", linewidth=1)
ax.text(len(bins) - 1, 1.15, "uniform value $1/N$", fontsize=6.5, color=INK2, ha="right", va="bottom")
ax.set_xticks(range(len(bins)))
ax.set_xticklabels(lab, rotation=45, ha="right")
ax.set_yscale("log")
ax.set_xlabel("in-degree (bins of doubling width)")
ax.set_ylabel("mean PageRank $\\times N$")
ax.set_title("PageRank against in-degree")
save(fig, "google")
sh = gg["rank_mass_share"]
num("ggTopA", sh["top_0.1pct"] * 100, "{:.1f}")
num("ggTopB", sh["top_1pct"] * 100, "{:.1f}")
num("ggTopC", sh["top_10pct"] * 100, "{:.1f}")
num("ggSpear", gg["spearman_pr_indeg"], "{:.2f}")
num("ggOverlap", gg["top100_overlap_pr_vs_indeg"], "{:d}")
num("ggZero", gg["zero_indeg"], "{:,d}")
nums["ggZero"] = nums["ggZero"].replace(",", "{,}")
num("ggZeroPct", gg["zero_indeg"] / gg["N"] * 100, "{:.0f}")
num("ggMinRel", gg["min_pr"] / gg["uniform_pr"], "{:.2f}")
num("ggMaxRel", gg["max_pr"] / gg["uniform_pr"], "{:.0f}")
num("ggRate", rs_[-1] / rs_[-2], "{:.3f}")
odd = [t for t in gg["top"] if t["indeg_rank"] > 2 * t["rank"]]
if odd:
    num("ggOddRank", odd[0]["rank"], "{:d}")
    num("ggOddDeg", odd[0]["indeg_rank"], "{:d}")

# ---------------------------------------------------------------- validation
val = {}
for f in ["val/val_cpu.csv", "val/val_gpu.csv"]:
    for r in csv.DictReader(open(f"{RES}/{f}")):
        val[(r["impl"], r["ranks"], r["part"], r["comm"])] = r
vs = {}
for f in glob.glob(f"{RES}/val/valscale*.csv"):
    for r in csv.DictReader(open(f)):
        vs[int(r["ranks"])] = r
cmp = {}
for f in ["val/compare_google.json", "val/compare_lj50.json"]:
    for k, v in json.load(open(f"{RES}/{f}"))["vectors"].items():
        cmp[os.path.basename(k)] = v


def sci(x):
    m, e = f"{x:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


with open(f"{GEN}/tab_val.tex", "w") as f:
    for b, lab in [("mpi_vertex_allgather.bin", "MPI+OpenMP, vertex partition, allgather"),
                   ("mpi_vertex_ghost.bin", "MPI+OpenMP, vertex partition, ghost"),
                   ("mpi_edge_allgather.bin", "MPI+OpenMP, edge partition, allgather"),
                   ("mpi_edge_ghost.bin", "MPI+OpenMP, edge partition, ghost"),
                   ("cuda_thread.bin", "CUDA, one thread per row"), ("cuda_warp.bin", "CUDA, one warp per row"),
                   ("cuda_hybrid.bin", "CUDA, degree buckets"), ("cuda_cusparse.bin", "CUDA, \\texttt{cusparseSpMV}"),
                   ("mpi_cuda_1gpu_hybrid.bin", "MPI+CUDA, 1 GPU"), ("mpi_cuda_2gpu_hybrid.bin", "MPI+CUDA, 2 GPUs")]:
        v = cmp[b]
        f.write(f"web-Google & {lab} & {sci(v['l1'])} & {sci(v['linf'])} & "
                f"{'yes' if v['top100_same_order'] else 'no'} \\\\\n")
    f.write("\\addlinespace[2pt]\n")
    for p in sorted(x for x in vs if x > 1):
        v = cmp[f"mpi_lj50_p{p}.bin"]
        f.write(f"soc-LiveJournal1 & pure MPI, {p} ranks, {p // 72} nodes & {sci(v['l1'])} & {sci(v['linf'])} & "
                f"{'yes' if v['top100_same_order'] else 'no'} \\\\\n")
# Every timed repetition records its final residual and the sum of the rank vector: count the rows, take the largest
# deviation of the sum from 1, and check that runs of the same graph and iteration count report the same residual
# (compared only where the residual is still above rounding level).
allrows = [r for f in sorted(glob.glob(f"{RES}/*.csv")) + sorted(glob.glob(f"{RES}/val/*.csv"))
           for r in csv.DictReader(open(f)) if r.get("exp") and r["exp"] != "exp" and r.get("sumr")]
num("valRows", len(allrows), "{:d}")
nums["valSumErr"] = sci(max(abs(float(r["sumr"]) - 1) for r in allrows))
resid = groups(allrows, "graph", "iters")
checked = {k: {r["resid"] for r in v} for k, v in resid.items() if min(float(r["resid"]) for r in v) > 1e-12}
assert all(len(v) == 1 for v in checked.values()), "runs of the same graph and iteration count disagree"
vg = {(r["part"], r["comm"]): float(r["t_iter"]) * 1e3 for r in csv.DictReader(open(f"{RES}/val/val_cpu.csv"))
      if r["impl"] == "mpi_omp"}
num("valGGghost", vg[("edge", "ghost")], "{:.2f}")
num("valGGallg", vg[("edge", "allgather")], "{:.2f}")
num("valGGgain", (1 - vg[("edge", "ghost")] / vg[("edge", "allgather")]) * 100, "{:.0f}")
nx_ = json.load(open(f"{RES}/val/validation_networkx.json"))["networkx"]
nums["valNxLone"] = sci(nx_["l1"])
num("valIters", int(val[("serial", "1", "none", "none")]["iters"]), "{:d}")
num("valScaleMax", max(p for p in vs if f"mpi_lj50_p{p}.bin" in cmp), "{:d}")
nums["valMaxLone"] = sci(max(v["l1"] for v in cmp.values()))
nums["valGoogleLone"] = sci(max(v["l1"] for k, v in cmp.items() if "lj50" not in k))
assert all(v["passed"] for v in cmp.values())

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
        assert b not in nums, f"{b} is used both as a number and as a prefix"
        f.write(f"\\newcommand{{\\{b}}}[1]{{\\csname {b}#1\\endcsname}}\n")
print(f"wrote {len(os.listdir(FIG))} figures and {len(nums)} numbers")
missing = [n for n, v in nums.items() if v == "n/a"]
if missing:
    print("missing numbers:", missing)
