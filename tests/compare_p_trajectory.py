import csv
from pathlib import Path
root=Path('sync_gpu/tests/fixtures/figure1_reference')
out=Path('results/figure1_cpp_gpu_faithful/parity'); out.mkdir(parents=True,exist_ok=True)
py=list(csv.DictReader((root/'low_rank_trajectory.csv').open()))
cu=list(csv.DictReader((out/'p_cpp_trajectory.csv').open()))
names=['gamma0','gamma1','gamma2','gamma3','gamma4','delta','alpha','beta']
with (out/'p_trajectory_parity.csv').open('w',newline='') as f:
 w=csv.writer(f); w.writerow(['iteration','quantity','index','python_value','cpp_value','abs_error','rel_error'])
 rows=[]
 for a,b in zip(py,cu[:len(py)]):
  it=int(a['iteration'])
  for i,n in enumerate(names):
   pn=n if n in a else {'gamma0':'gamma_0','gamma1':'gamma_1','gamma2':'gamma_2','gamma3':'gamma_3','gamma4':'gamma_4'}[n]
   x=float(a[pn]); y=float(b[n]); e=abs(x-y); w.writerow([it,n,i,repr(x),repr(y),repr(e),repr(e/max(1e-300,abs(x)))])
   rows.append((it,n,e))
with (out/'p_trajectory_error_by_iteration.csv').open('w',newline='') as f:
 w=csv.writer(f); w.writerow(['iteration','gamma_max_abs_error','delta_abs_error','alpha_abs_error','beta_abs_error','state_max_abs_error','accepted_step_match','halving_count_match'])
 for it in range(1,len(py)):
  rr=[x for x in rows if x[0]==it]; vals={q:e for _,q,e in rr}; w.writerow([it,max(vals.get(n,0) for n in names[:5]),vals.get('delta',0),vals.get('alpha',0),vals.get('beta',0),max(vals.values()),'unavailable','unavailable'])
print('PY_RECORDS',len(py)-1,'CPP_RECORDS',len(cu)-1)
for it in [1,2,5,10,25,50]: print('CHECKPOINT',it,max(e for k,n,e in rows if k==it))
for n in names: print('MAX',n,max(e for k,q,e in rows if q==n))
print('OVERALL',max(e for _,_,e in rows),max(rows,key=lambda x:x[2]))
