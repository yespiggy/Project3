"""Compare fixed-iteration native float64 output vectors; standard library only."""
import array,json,math,struct,sys
from pathlib import Path
graph,folder=sys.argv[1:];folder=Path(folder)
with open(graph,'rb') as f:
    assert f.read(8)==b'PRGRAPH1'
    n,m=struct.unpack('qq',f.read(16))
def read(p):
    assert p.stat().st_size==8*n, f'Wrong vector size: {p}'
    a=array.array('d');a.frombytes(p.read_bytes())
    assert all(math.isfinite(x) and x>=0 for x in a),f'Invalid values: {p}'
    return a
ref=read(folder/'serial.bin');rsum=math.fsum(ref)
checks={}
for p in sorted(folder.glob('mpi_*_*.bin')):
    a=read(p);l1=math.fsum(abs(x-y) for x,y in zip(a,ref));linf=max(abs(x-y) for x,y in zip(a,ref));total=math.fsum(a)
    checks[p.name]={'l1':l1,'linf':linf,'sum':total,'passed':l1<=1e-8 and linf<=1e-9 and abs(total-1)<=1e-8}
passed=len(checks)==4 and abs(rsum-1)<=1e-8 and all(v['passed'] for v in checks.values())
out={'comparison':'same graph, same fixed 20 iterations; not convergence validation','N':n,'serial_sum':rsum,'tolerances':{'l1':1e-8,'linf':1e-9,'sum_error':1e-8},'vectors':checks,'passed':passed}
(folder/'validation.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if not passed:sys.exit('Full-vector validation failed')
