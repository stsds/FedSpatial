#!/usr/bin/env python3
"""Aggregate only completed fresh P-only two-worker Figure-1 cases."""
import argparse, csv, json, math
from pathlib import Path
def main():
 p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--n',type=int,default=4000); p.add_argument('--knots',type=int,default=400); a=p.parse_args(); rows=[]; bad=[]
 for s in range(9):
  for r in range(20):
   n=a.root/'cases'/f'scenario_{s:02d}'/f'rep_{r:03d}'/'native'
   try:
    m=json.loads((n/'launch_manifest.json').read_text()); q=list(csv.DictReader((n/'native_estimates.csv').open()))
    ok=m.get('scenario_id')==s and m.get('replication')==r and m.get('n')==a.n and m.get('m')==a.knots and m.get('q')==5 and m.get('workers')==2 and m.get('worker_rows')==a.n//2 and m.get('rounds')==50 and m.get('p_only') is True and m.get('local_solve_strategy')=='streamed_trsm' and m.get('dense_w_enabled') is False and m.get('shared_global_knots') is True and m.get('producer_tile_width')==400 and len(q)==1 and q[0].get('method')=='P' and q[0].get('status')=='OK'
    vals=[float(q[0][x]) for x in ('gamma0','gamma1','gamma2','gamma3','gamma4','delta','alpha','beta','runtime_seconds')]
    if not ok or not all(math.isfinite(x) for x in vals): raise ValueError('settings/output validation')
    rows.append({'scenario_id':s,'replication':r,**q[0]})
   except Exception as e: bad.append({'scenario_id':s,'replication':r,'error':str(e)})
 report={'expected_cases':180,'complete_cases':len(rows),'missing_or_invalid':bad,'status':'PASS' if len(rows)==180 and not bad else 'FAIL'}
 (a.root/'completeness.json').write_text(json.dumps(report,indent=2)+'\n')
 if rows:
  with (a.root/'native_p_estimates.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
 print(json.dumps(report)); raise SystemExit(0 if report['status']=='PASS' else 1)
if __name__=='__main__': main()
