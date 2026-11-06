"""Optional figures: python3 summarize.py && python3 plot.py (matplotlib)."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;O=R/'figures';O.mkdir(exist_ok=True)
def rows(n):
 with (R/'tables'/n).open() as f:return list(csv.DictReader(f))
def error(ax,rs,label,**kw):
 x=[int(r['cores']) for r in rs];y=[float(r['median_ms']) for r in rs]
 ax.errorbar(x,y,yerr=[[v-float(r['min_ms']) for v,r in zip(y,rs)],[float(r['max_ms'])-v for v,r in zip(y,rs)]],label=label,capsize=3,**kw)
s=rows('cpu_rmat_strong_and_layout.csv');fig,ax=plt.subplots(figsize=(8,5))
error(ax,[r for r in s if r['exp']=='E1_hybrid' and r['batch']!='1512555_partial_hybrid'],'Hybrid: 1–72 and 256 cores',marker='o')
error(ax,[r for r in s if r['exp']=='E1_mpi'],'Pure MPI (720-rank run failed)',marker='s')
for b,label in [('1512555_partial_hybrid','720 cores, earlier 20×36 batch'),('1515216','720 cores, new 20×36 batch')]:
 error(ax,[r for r in s if r['batch']==b and r['ranks']=='20'],label,marker='D',linestyle='none')
ax.set(xscale='log',yscale='log',xlabel='Active CPU cores',ylabel='Median iteration time (ms)',title='R-MAT strong scaling: same graph, 20 iterations')
ax.legend(fontsize=8);ax.grid(alpha=.25);fig.tight_layout();fig.savefig(O/'cpu_strong.png',dpi=180);plt.close(fig)
d=[r for r in s if r['batch']=='1515216'];fig,ax=plt.subplots(figsize=(7,4))
ax.bar([r['ranks']+' × '+r['threads'] for r in d],[float(r['median_ms']) for r in d]);ax.set(xlabel='MPI ranks × OpenMP threads (720 total cores)',ylabel='Median iteration time (ms)',title='720-core layout comparison: job 1515216')
fig.tight_layout();fig.savefig(O/'cpu_720_layout.png',dpi=180);plt.close(fig)
w=rows('cpu_rmat_weak.csv');fig,ax=plt.subplots(figsize=(7,4))
ax.plot([int(r['cores']) for r in w],[100*float(r['nominal_weak_efficiency']) for r in w],'o-');ax.set(xscale='log',xlabel='Active CPU cores',ylabel='Nominal weak efficiency (%)',title='Approximate R-MAT weak scaling')
ax.text(.02,.03,'256-core point used GN-A40 CPU nodes; topology/binding unverified.\nRealized edges/core rise by 6.29% from 1 to 256 cores.',transform=ax.transAxes,fontsize=8);ax.grid(alpha=.25)
fig.tight_layout();fig.savefig(O/'cpu_weak.png',dpi=180);plt.close(fig)
g=rows('gpu_scaling.csv');fig,axs=plt.subplots(1,2,figsize=(10,4))
for ax,graph in zip(axs,['soc-LiveJournal1','com-friendster']):
 for batch,label in [('1508084_same_node','Same node'),('1508085_cross_node','Across nodes')]:
  rs=[r for r in g if r['batch']==batch and r['graph']==graph and r['impl']=='mpi_cuda_hybrid'];ax.plot([int(r['ranks']) for r in rs],[float(r['speedup_vs_1gpu']) for r in rs],'o-',label=label)
 ax.plot([1,2],[1,2],'k--',alpha=.4,label='Ideal');ax.set(title=graph,xlabel='GPUs',ylabel='Speedup vs matching 1-GPU run',xticks=[1,2]);ax.legend(fontsize=8);ax.grid(alpha=.25)
fig.suptitle('MPI+CUDA hybrid kernel: provisional until full-vector validation');fig.tight_layout();fig.savefig(O/'gpu_scaling.png',dpi=180);plt.close(fig)
