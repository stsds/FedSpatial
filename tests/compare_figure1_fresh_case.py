#!/usr/bin/env python3
"""Compare fresh Python and native Figure-1 trajectories without estimate inputs."""
import argparse,csv,json,math
from pathlib import Path
def load(path):
    with path.open() as f:return {(r['method'],r['iteration']):r for r in csv.DictReader(f)}
def main():
 p=argparse.ArgumentParser();p.add_argument('--python-dir',type=Path,required=True);p.add_argument('--native-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--abs-tolerance',type=float,default=1e-5);p.add_argument('--relative-tolerance',type=float,default=1e-5);a=p.parse_args();py=load(a.python_dir/'python_trajectory.csv');na=load(a.native_dir/'native_trajectory.csv');fields=['gamma0','gamma1','gamma2','gamma3','gamma4','delta','alpha','beta'];report={'absolute_tolerance':a.abs_tolerance,'relative_tolerance_unit_floor':a.relative_tolerance};passed=True
 for method in ('P','IND'):
  absmax={x:0. for x in fields};relmax={x:0. for x in fields};keys=[(method,str(i)) for i in range(51)]
  if any(k not in py or k not in na for k in keys):raise SystemExit(f'{method}: expected all 51 state checkpoints')
  growth=[];step_mismatch=[]
  for k in keys:
   for x in fields:
    d=abs(float(py[k][x])-float(na[k][x]));absmax[x]=max(absmax[x],d);relmax[x]=max(relmax[x],d/max(1.,abs(float(py[k][x]))))
   growth.append(max(abs(float(py[k][x])-float(na[k][x])) for x in fields))
   if abs(float(py[k]['accepted_step'])-float(na[k]['accepted_step']))>1e-15 or int(float(py[k]['halving_count']))!=int(float(na[k]['halving_count'])):step_mismatch.append(k[1])
  final={x:abs(float(py[(method,'50')][x])-float(na[(method,'50')][x])) for x in fields}
  report[method]={'checkpoints':len(keys),'max_abs':absmax,'max_rel_unit_floor':relmax,'iteration_max_abs':growth,'final_state_abs':final,'plotted_final_alpha_abs':final['alpha'],'plotted_final_beta_abs':final['beta'],'step_halving_mismatch_iterations':step_mismatch};passed &= max(absmax.values())<=a.abs_tolerance and max(relmax.values())<=a.relative_tolerance and not step_mismatch
 report['status']='PASS' if passed else 'FAIL';a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));raise SystemExit(0 if passed else 1)
if __name__=='__main__':main()
