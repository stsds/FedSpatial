#!/usr/bin/env python3
"""Restartable, one-case-at-a-time Figure-1 P-only two-GV100 supervisor."""
import argparse, csv, json, os, subprocess, sys, threading, time
from datetime import datetime, timezone
from pathlib import Path

UUIDS='GPU-737f3c7a-bf62-20b7-01f8-b3fb1a3c86f4,GPU-a7c40d19-eb27-1d13-257a-5949f98fc947'
GATES=('FEDSPATIAL_MIXED_TRACE_TILED_ACTION','FEDSPATIAL_MIXED_CONTRACTIONS_TILED','FEDSPATIAL_THETA_FIRST_TILED','FEDSPATIAL_THETA_HESSIAN_AA_LOG_TILED','FEDSPATIAL_THETA_HESSIAN_AR_LOG_TILED','FEDSPATIAL_THETA_HESSIAN_RR_LOG_TILED','FEDSPATIAL_THETA_HESSIAN_Q_W_TILED','FEDSPATIAL_THETA_HESSIAN_Q_PANEL_TILED','FEDSPATIAL_THETA_HESSIAN_SECOND_LOG_TILED','FEDSPATIAL_THETA_HESSIAN_Q_BASE_REUSE','FEDSPATIAL_THETA_HESSIAN_Q_SUPERPANEL','FEDSPATIAL_THETA_SECOND_TRIPLE_BATCHED')

def now(): return datetime.now(timezone.utc).isoformat()
def atom(path, value):
    tmp=path.with_name(path.name+f'.tmp.{os.getpid()}'); tmp.write_text(json.dumps(value,indent=2)+'\n'); os.replace(tmp,path)
def fixture_ok(path, scenario, rep, n, knots, workers):
    try:
        m=json.loads((path/'metadata.json').read_text())
        want={'scenario_id':scenario,'replication':rep,'seed':rep+1,'n':n,'p':5,'knots':knots,'workers':workers,'observations_per_worker':n//workers,'partition_order':'global_then_consecutive'}
        return all(m.get(k)==v for k,v in want.items())
    except Exception: return False
def complete(path, scenario, rep, n, knots, workers, rounds):
    try:
        launch=json.loads((path/'launch_manifest.json').read_text())
        required={'scenario_id':scenario,'replication':rep,'seed':rep+1,'n':n,'m':knots,'q':5,'workers':workers,'worker_rows':n//workers,'rounds':rounds,'p_only':True,'uuid_mask':UUIDS,'local_solve_strategy':'streamed_trsm','dense_w_enabled':False,'shared_global_knots':True,'producer_tile_width':400,'optimized_m400_gates':list(GATES)}
        if any(launch.get(k)!=v for k,v in required.items()): return False
        rows=list(csv.DictReader((path/'native_estimates.csv').open()))
        values=[float(rows[0][x]) for x in ('gamma0','gamma1','gamma2','gamma3','gamma4','delta','alpha','beta','runtime_seconds')]
        return len(rows)==1 and rows[0].get('method')=='P' and rows[0].get('status')=='OK' and all(__import__('math').isfinite(x) for x in values) and len(list(csv.DictReader((path/'native_round_timing.csv').open())))==rounds
    except Exception: return False
def env():
    out=os.environ.copy(); out['CUDA_VISIBLE_DEVICES']=UUIDS; out['FEDSPATIAL_STAGE_CUDA_TIMING']='1'; out['FEDSPATIAL_LOCAL_SOLVE_STRATEGY']='streamed_trsm'; out['FEDSPATIAL_PRODUCER_TILE_WIDTH']='400'
    for gate in GATES: out[gate]='1'
    return out
def telemetry(stop, path):
    query='uuid,pci.bus_id,utilization.gpu,utilization.memory,memory.used,memory.free'
    with path.open('w',buffering=1) as f:
        f.write('timestamp_utc,uuid,pci_bus,gpu_pct,mem_pct,used_mib,free_mib\n')
        while not stop.wait(5):
            try:
                for row in subprocess.check_output(['nvidia-smi',f'--query-gpu={query}','--format=csv,noheader,nounits'],text=True).splitlines(): f.write(now()+','+row.replace(', ',',')+'\n')
            except Exception as error: f.write(now()+',telemetry_error,'+type(error).__name__+'\n')
def execute(cmd, stdout, stderr, envv, telemetry_path, timeout=7200):
    stop=threading.Event(); thread=threading.Thread(target=telemetry,args=(stop,telemetry_path),daemon=True); thread.start()
    try:
        return subprocess.run(['timeout','--preserve-status',f'{timeout}s','stdbuf','-oL','-eL',*map(str,cmd)],stdout=stdout,stderr=stderr,env=envv).returncode
    finally: stop.set(); thread.join(6)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--build-dir',type=Path,required=True); ap.add_argument('--output-root',type=Path,required=True); ap.add_argument('--scenario-start',type=int,default=0); ap.add_argument('--scenario-end',type=int,default=8); ap.add_argument('--rep-start',type=int,default=0); ap.add_argument('--rep-end',type=int,default=19); ap.add_argument('--n',type=int,default=4000); ap.add_argument('--knots',type=int,default=400); ap.add_argument('--workers',type=int,default=2); ap.add_argument('--rounds',type=int,default=50); ap.add_argument('--resume',action='store_true'); args=ap.parse_args()
    root=Path(__file__).resolve().parents[2]; exporter=root/'sync_gpu/tests/export_figure1_python_fixture.py'; exe=args.build_dir/'sync_gpu_figure1_fresh_case'
    if not exe.is_file(): raise SystemExit('missing executable')
    if (args.n,args.knots,args.workers,args.rounds)!=(4000,400,2,50): raise SystemExit('this supervisor is restricted to the validated 2x2000, m=400, 50-round target')
    args.output_root.mkdir(parents=True,exist_ok=True); lock=args.output_root/'supervisor.lock'
    try: os.mkdir(lock)
    except FileExistsError: raise SystemExit('supervisor already active')
    state=args.output_root/'supervisor_status.json'; atom(state,{'status':'RUNNING','start_utc':now(),'uuid_mask':UUIDS,'p_only':True,'n':args.n,'m':args.knots,'workers':args.workers,'rounds':args.rounds,'local_solve_strategy':'streamed_trsm','dense_w_enabled':False,'producer_tile_width':400,'optimized_m400_gates':list(GATES),'completed':0,'skipped':0})
    completed=skipped=0
    try:
      for s in range(args.scenario_start,args.scenario_end+1):
       for r in range(args.rep_start,args.rep_end+1):
        case=args.output_root/'cases'/f'scenario_{s:02d}'/f'rep_{r:03d}'; native=case/'native'; raw=args.output_root/'raw'/f'scenario_{s:02d}'/f'rep_{r:03d}'
        if args.resume and complete(native,s,r,args.n,args.knots,args.workers,args.rounds): skipped+=1; atom(state,{'status':'RUNNING','last_case':[s,r],'completed':completed,'skipped':skipped,'updated_utc':now()}); continue
        case.mkdir(parents=True,exist_ok=True)
        if not fixture_ok(raw,s,r,args.n,args.knots,args.workers):
            subprocess.run([sys.executable,exporter,'--scenario-id',str(s),'--replication',str(r),'--output-root',args.output_root/'raw','--n',str(args.n),'--knots',str(args.knots),'--workers',str(args.workers)],check=True,stdout=(case/'fixture_stdout.log').open('w'),stderr=(case/'fixture_stderr.log').open('w'))
        if not fixture_ok(raw,s,r,args.n,args.knots,args.workers): raise RuntimeError(f'fixture preflight failed {s}/{r}')
        tmp=case/f'native.tmp.{os.getpid()}'
        if tmp.exists(): raise RuntimeError(f'stale temporary {tmp}')
        tmp.mkdir(); command=[exe,'--fixture-root',args.output_root/'raw','--output-dir',tmp,'--scenario-id',str(s),'--replication',str(r),'--worker-devices','1,2','--workers',str(args.workers),'--rounds',str(args.rounds),'--p-only','--two-worker-2k-m400-production']
        (tmp/'launch_command.txt').write_text(' '.join(map(str,command))+'\n'); atom(tmp/'launch_manifest.json',{'scenario_id':s,'replication':r,'seed':r+1,'n':args.n,'m':args.knots,'q':5,'workers':args.workers,'worker_rows':args.n//args.workers,'rounds':args.rounds,'p_only':True,'uuid_mask':UUIDS,'local_solve_strategy':'streamed_trsm','dense_w_enabled':False,'shared_global_knots':True,'producer_tile_width':400,'optimized_m400_gates':list(GATES),'start_utc':now()})
        with (tmp/'stdout.log').open('w',buffering=1) as out,(tmp/'stderr.log').open('w',buffering=1) as err:
            rc=execute(command,out,err,env(),tmp/'telemetry.csv')
        (tmp/'exit_code').write_text(str(rc)+'\n')
        if rc or not complete(tmp,s,r,args.n,args.knots,args.workers,args.rounds): atom(case/'FAILED.json',{'scenario_id':s,'replication':r,'exit_code':rc,'error':'native case failed validation','end_utc':now()}); raise RuntimeError(f'case failed {s}/{r}')
        if native.exists(): raise RuntimeError(f'refusing overwrite {native}')
        os.replace(tmp,native); atom(native/'COMPLETE.json',{'scenario_id':s,'replication':r,'status':'OK','end_utc':now()}); completed+=1; atom(state,{'status':'RUNNING','last_case':[s,r],'completed':completed,'skipped':skipped,'updated_utc':now()})
      atom(state,{'status':'PASS','completed':completed,'skipped':skipped,'end_utc':now()})
    except Exception as error:
      atom(state,{'status':'FAIL','completed':completed,'skipped':skipped,'error':str(error),'end_utc':now()}); raise
    finally: os.rmdir(lock)
if __name__=='__main__': main()
