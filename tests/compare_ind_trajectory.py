import csv
from pathlib import Path
root=Path('sync_gpu/tests/fixtures/figure1_reference'); out=Path('results/figure1_cpp_gpu_faithful/parity')
py=list(csv.DictReader((root/'ind_trajectory.csv').open())); cu=list(csv.DictReader((out/'ind_cpp_trajectory.csv').open()))
names=['gamma0','gamma1','gamma2','gamma3','gamma4','delta','alpha','beta']; rows=[]
with (out/'ind_trajectory_parity.csv').open('w',newline='') as f:
 w=csv.writer(f); w.writerow(['iteration','quantity','index','python_value','cpp_value','abs_error','rel_error'])
 for a,b in zip(py,cu):
  it=int(a['iteration'])
  for i,n in enumerate(names):
   pn=n if n in a else {'gamma0':'gamma_0','gamma1':'gamma_1','gamma2':'gamma_2','gamma3':'gamma_3','gamma4':'gamma_4'}[n]; x=float(a[pn]); y=float(b[n]); e=abs(x-y); rows.append((it,n,e)); w.writerow([it,n,i,x,y,e,e/max(1e-300,abs(x))])
with (out/'ind_trajectory_error_by_iteration.csv').open('w',newline='') as f:
 w=csv.writer(f); w.writerow(['iteration','gamma_max_abs_error','delta_abs_error','alpha_abs_error','beta_abs_error','state_max_abs_error'])
 for it in range(1,51):
  v={q:e for k,q,e in rows if k==it}; w.writerow([it,max(v[n] for n in names[:5]),v['delta'],v['alpha'],v['beta'],max(v.values())])
print('records_python',len(py)-1,'records_cpp',len(cu)-1)
for it in [1,2,5,10,25,50]: print('checkpoint',it,max(e for k,q,e in rows if k==it))
for n in names: print('max',n,max(e for k,q,e in rows if q==n))
print('overall',max(rows,key=lambda x:x[2]))
with (out/'ind_global_newton_parity.csv').open('w',newline='') as f:
 w=csv.writer(f); w.writerow(['quantity','python_value','cpp_value','abs_error'])
 a,b=py[1],cu[1]
 for n in ['delta','alpha','beta']:
  w.writerow([n,float(a[n]),float(b[n]),abs(float(a[n])-float(b[n]))])
