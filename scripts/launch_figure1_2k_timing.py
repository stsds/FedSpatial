#!/usr/bin/env python3
"""One detached, UUID-masked 2x2000 Figure-1 P timing round."""
import json, os, subprocess, sys, threading, time
from datetime import datetime, timezone
from pathlib import Path

UUIDS="GPU-737f3c7a-bf62-20b7-01f8-b3fb1a3c86f4,GPU-a7c40d19-eb27-1d13-257a-5949f98fc947"
GATES=("FEDSPATIAL_MIXED_TRACE_TILED_ACTION","FEDSPATIAL_MIXED_CONTRACTIONS_TILED","FEDSPATIAL_THETA_FIRST_TILED","FEDSPATIAL_THETA_HESSIAN_AA_LOG_TILED","FEDSPATIAL_THETA_HESSIAN_AR_LOG_TILED","FEDSPATIAL_THETA_HESSIAN_RR_LOG_TILED","FEDSPATIAL_THETA_HESSIAN_Q_W_TILED","FEDSPATIAL_THETA_HESSIAN_Q_PANEL_TILED","FEDSPATIAL_THETA_HESSIAN_SECOND_LOG_TILED")
def now(): return datetime.now(timezone.utc).isoformat()
def put(p,x): p.write_text(json.dumps(x,indent=2)+"\n")
def telemetry(stop,p):
  q="uuid,pci.bus_id,utilization.gpu,utilization.memory,memory.used,memory.free"
  with p.open("w",buffering=1) as f:
    f.write("timestamp_utc,uuid,pci_bus,gpu_pct,mem_pct,used_mib,free_mib\n")
    while not stop.wait(3):
      try:
        for line in subprocess.check_output(["nvidia-smi",f"--query-gpu={q}","--format=csv,noheader,nounits"],text=True).splitlines(): f.write(now()+","+line.replace(", ",",")+"\n")
      except Exception as e: f.write(now()+",telemetry_error,"+type(e).__name__+"\n")
def main():
  if len(sys.argv)!=4: raise SystemExit("usage: OUTPUT EXECUTABLE FIXTURE_LEAF")
  out,exe,fixture=map(Path,sys.argv[1:]); out.mkdir(parents=True,exist_ok=False)
  meta=json.loads((fixture/"metadata.json").read_text())
  want={"scenario_id":4,"replication":0,"seed":1,"n":4000,"p":5,"knots":2000,"workers":2,"observations_per_worker":2000,"partition_order":"global_then_consecutive"}
  bad={k:(meta.get(k),v) for k,v in want.items() if meta.get(k)!=v}
  if bad: raise SystemExit("invalid 2k target fixture: "+repr(bad))
  env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=UUIDS;env["FEDSPATIAL_STAGE_CUDA_TIMING"]="1"
  for g in GATES: env[g]="1"
  cmd=["timeout","--preserve-status","1800s","stdbuf","-oL","-eL",str(exe),"--fixture-root",str(fixture.parent.parent),"--output-dir",str(out),"--scenario-id","4","--replication","0","--worker-devices","1,2","--workers","2","--rounds","1","--p-only","--two-worker-2k-timing-target","--one-round-timing"]
  put(out/"launch_manifest.json",{"kind":"authorized_2x2000_one_round","start_utc":now(),"fixture":str(fixture),"uuid_mask":UUIDS,"workers":2,"n_per_worker":2000,"total_n":4000,"m":2000,"q":5,"fp":"FP64","rounds":1,"watchdog_seconds":1800,"gates":list(GATES),"fixture_metadata":meta})
  (out/"launch_command.txt").write_text(" ".join(cmd)+"\n")
  put(out/"status.json",{"status":"STARTING","start_utc":now()});stop=threading.Event();t=threading.Thread(target=telemetry,args=(stop,out/"telemetry.csv"),daemon=True)
  with (out/"stdout.log").open("w",buffering=1) as stdout, (out/"stderr.log").open("w",buffering=1) as stderr:
    p=subprocess.Popen(cmd,stdout=stdout,stderr=stderr,env=env,start_new_session=True);(out/"pid.txt").write_text(str(p.pid)+"\n");put(out/"status.json",{"status":"RUNNING","pid":p.pid,"start_utc":now()});t.start();code=p.wait()
  stop.set();t.join(5);put(out/"status.json",{"status":"PASS" if code==0 else ("WATCHDOG" if code==124 else "FAIL"),"pid":p.pid,"exit_code":code,"end_utc":now()})
if __name__=="__main__": main()
