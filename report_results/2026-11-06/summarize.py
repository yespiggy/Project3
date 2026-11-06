"""Rebuild tables from explicit source files; no historical/archive double counting."""
from pathlib import Path
import csv,statistics,json,math
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'tables';OUT.mkdir(exist_ok=True)
def read(path):
 with (ROOT/path).open() as f:
  rows=list(csv.DictReader(f))
 for r in rows:
  if None in r:
   if any(x for x in r[None]): raise ValueError('unexpected extra columns')
   del r[None]  # legacy file appended an empty t_stage field
 return rows
def summarize(paths):
 output=[]
 for batch,path in paths:
  groups={}
  for r in read(path):
   key=tuple(r[k] for k in ('exp','impl','graph','N','M','nodes','ranks','threads','cores','part','comm','iters'))
   groups.setdefault(key,[]).append(r)
  for key,rs in groups.items():
   assert len(set(r['rep'] for r in rs))==len(rs),(batch,key,'duplicate repetition')
   x=dict(zip(('exp','impl','graph','N','M','nodes','ranks','threads','cores','part','comm','iters'),key))
   times=[float(r['t_iter'])*1000 for r in rs];assert all(math.isfinite(t) and t>0 for t in times)
   x.update(batch=batch,source=path,repetitions=len(rs),median_ms=statistics.median(times),min_ms=min(times),max_ms=max(times))
   for c in ('t_comp','t_comm','t_stage'):
    vals=[float(r[c])/int(r['iters'])*1000 for r in rs if r.get(c)]
    x[c+'_median_ms']=statistics.median(vals) if vals else ''
   x['edges_per_core']=int(x['M'])/int(x['cores'])
   output.append(x)
 return output
def write(name,rs):
 keys=list(dict.fromkeys(k for r in rs for k in r))
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rs)
strong=summarize([('1509305','raw/ours/results/strong_n1_synthetic.csv'),('1518822','raw/ours/results/strong_n4_c256_1518822.stdct-mgmt-02.csv'),('1512555_partial_hybrid','raw/ours/results/strong_n10_synthetic_1512555.stdct-mgmt-02.csv'),('1515216','raw/diagnostic_1515216/runs.csv')])
base=next(x['median_ms'] for x in strong if x['cores']=='1')
for x in strong:x.update(speedup_vs_1core=base/x['median_ms'],efficiency=base/x['median_ms']/int(x['cores']))
write('cpu_rmat_strong_and_layout.csv',strong)
weak=summarize([('1509305','raw/ours/results/weak_n1.csv'),('1521041_user_transcribed','raw/user_transcribed_1521041/weak_n4_c256.csv')])
b=next(x for x in weak if x['cores']=='1')
for x in weak:
 x.update(nominal_weak_efficiency=b['median_ms']/x['median_ms'],edges_per_core_ratio=x['edges_per_core']/b['edges_per_core'])
 x['caveat']='GN-A40 CPU model/binding unverified' if x['cores']=='256' else 'CN-node CPU run'
write('cpu_rmat_weak.csv',weak)
gpu=summarize([('1508084_same_node','raw/teammate/results/multigpu_1node.csv'),('1508085_cross_node','raw/teammate/results/multigpu_2node.csv')])
for x in gpu:
 b=next(y for y in gpu if y['batch']==x['batch'] and y['graph']==x['graph'] and y['impl']==x['impl'] and y['ranks']=='1')
 x.update(speedup_vs_1gpu=b['median_ms']/x['median_ms'],gpu_efficiency=b['median_ms']/x['median_ms']/int(x['ranks']))
 x['caveat']='multi-GPU full-vector verification and exact source revision missing; MPI wall-time definition requires confirmation'
write('gpu_scaling.csv',gpu)
lj=summarize([('1508081_cpu','raw/teammate/results/strong_n1_lj.csv'),('1508081_serial','raw/teammate/results/serial.csv')])
write('cpu_livejournal.csv',lj)
# E4 contains two appended blocks with repeated rep IDs. Keep both, with unknown provenance explicit.
e4=read('raw/teammate/results/e4.csv');assert len(e4)==40
norm=ROOT/'normalized';norm.mkdir(exist_ok=True)
for i in (0,1):
 with (norm/f'e4_block{i+1}.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(e4[0]));w.writeheader();w.writerows(e4[i*20:(i+1)*20])
e4s=summarize([(f'e4_block{i}_provenance_unconfirmed',f'normalized/e4_block{i}.csv') for i in (1,2)])
write('single_gpu_kernels_separate_batches.csv',e4s)
# Same graph only, and matched iteration counts. Distinct MPI+CUDA and standalone CUDA programs.
serial=next(x['median_ms'] for x in lj if x['impl']=='serial')
compare=[]
for x in lj+[x for x in gpu if x['graph']=='soc-LiveJournal1']+e4s:
 y=dict(x);y['speedup_vs_livejournal_serial']=serial/y['median_ms'];compare.append(y)
write('livejournal_cpu_gpu_comparison.csv',compare)
assert len(strong)==19 and len(weak)==8 and len(gpu)==16
print('Verified: 19 CPU strong/layout groups; 8 weak groups; 16 GPU groups; separate E4 batches.')

# Historical design ablation remains useful, but is never merged with updated timings.
historical=summarize([('legacy_e3_revision_unconfirmed','raw/teammate/results/e3.csv')])
for x in historical:x['caveat']='Historical batch; exact revision and timing semantics unconfirmed'
write('historical_design_choices.csv',historical)
print(f'Archived design comparison: {len(historical)} groups, explicitly historical.')
