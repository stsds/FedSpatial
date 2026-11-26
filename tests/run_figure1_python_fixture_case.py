#!/usr/bin/env python3
"""Run original Figure-1 Python estimators on one exported raw fixture."""
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'new_bersion_of_python_code'/'Asyn_Fed_Spatial'))
from src.kernel import exponential_kernel, onedif_kernel, Matern_2_5_kernel
from src.estimation_syn import GlobalComputation as PGlobal, LocalComputation as PLocal, Parameter as PParameter, Server as PServer, Worker as PWorker
from src.estimation_ind_syn import GlobalComputation as IGlobal, LocalComputation as ILocal, Parameter as IParameter, Server as IServer, Worker as IWorker

def values(param):
    return [float(x) for x in param.gamma.reshape(-1)]+[float(param.delta),float(param.theta[0]),float(param.theta[1])]
def write(writer,method,iteration,param,step,halves): writer.writerow([method,iteration,*values(param),step,halves])
def step_metadata(global_computation, param, quantity, p_mode):
    gradient=quantity['grad'].unsqueeze(1); hessian=quantity['hessian']
    if p_mode:
        correction=global_computation.Grad_Hessian_f(param); d=correction['grad'].numel()
        gradient=gradient*global_computation.global_para_ind_quantities['J']
        gradient=gradient+torch.cat([torch.zeros((1,1),dtype=torch.float64),correction['grad']],dim=0)
        hessian=hessian*global_computation.global_para_ind_quantities['J']
        padded=torch.zeros((d+1,d+1),dtype=torch.float64); padded[1:,1:]=correction['hessian']; hessian=hessian+padded
    eigvals,eigvecs=torch.linalg.eigh(hessian); modified=eigvecs@torch.diag(torch.maximum(eigvals.abs(),torch.tensor(.01,dtype=torch.float64)))@eigvecs.T
    direction=torch.linalg.inv(modified)@gradient; step=float(global_computation.step_size); halves=0
    candidate=torch.cat((param.delta.reshape(1),param.theta.reshape(-1)))-step*direction.reshape(-1)
    while (candidate<=0).any(): step/=2.; halves+=1; candidate=torch.cat((param.delta.reshape(1),param.theta.reshape(-1)))-step*direction.reshape(-1)
    return step,halves
def main():
    p=argparse.ArgumentParser();p.add_argument('--fixture-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    meta=json.loads((a.fixture_dir/'metadata.json').read_text()); nu,beta=meta['nu'],meta['beta']; kernel_fn={.5:exponential_kernel,1.5:onedif_kernel,2.5:Matern_2_5_kernel}[nu]
    theta=torch.tensor([1.,beta],dtype=torch.float64)
    def kernel(x,y=None,theta_arg=None):
        return kernel_fn(x,y,theta=theta if theta_arg is None else theta_arg)
    sites=torch.from_numpy(np.load(a.fixture_dir/'sites.npy'));X=torch.from_numpy(np.load(a.fixture_dir/'X.npy'));z=torch.from_numpy(np.load(a.fixture_dir/'z.npy')).reshape(-1,1);knots=torch.from_numpy(np.load(a.fixture_dir/'knots.npy'))
    data=torch.hstack((sites,z,X)); chunks=list(torch.chunk(data,10,dim=0));gamma=torch.tensor([-1,2,3,-2,1],dtype=torch.float64).reshape(-1,1);delta=torch.tensor(.25,dtype=torch.float64);param0={'gamma':gamma,'delta':delta,'theta':theta.reshape(-1,1)};a.output_dir.mkdir(parents=True,exist_ok=True)
    with (a.output_dir/'python_trajectory.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['method','iteration','gamma0','gamma1','gamma2','gamma3','gamma4','delta','alpha','beta','accepted_step','halving_count'])
        start=time.time(); workers=[PWorker(i+1,PLocal(x,kernel,knots)) for i,x in enumerate(chunks)];server=PServer(workers,PGlobal(knots,kernel,1.),type_LR='P',dl_th_together=True,if_log=False,logger=None); p_steps=[]; original=server.global_computation.update_delta_theta
        def p_update(param,quantity): p_steps.append(step_metadata(server.global_computation,param,quantity,True)); return original(param,quantity)
        server.global_computation.update_delta_theta=p_update; server.run(PParameter(**param0),T=50,method='direct');psec=time.time()-start
        for i,param in enumerate(server.params_GDT):write(w,'P',i,param,*p_steps[i])
        write(w,'P',50,server.param,0.,0); p_final=server.param
        start=time.time(); workers=[IWorker(i+1,ILocal(x,kernel,knots)) for i,x in enumerate(chunks)];server=IServer(workers,IGlobal(kernel,1.,'P'),type_LR='P',if_log=False,logger=None); i_steps=[]; original=server.global_computation.update_delta_theta
        def i_update(param,quantity): i_steps.append(step_metadata(server.global_computation,param,quantity,False)); return original(param,quantity)
        server.global_computation.update_delta_theta=i_update; server.run(IParameter(**param0),T=50,method='direct');isec=time.time()-start
        for i,param in enumerate(server.params):write(w,'IND',i,param,*i_steps[i])
        write(w,'IND',50,server.param,0.,0); i_final=server.param
    with (a.output_dir/'python_estimates.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['method','gamma0','gamma1','gamma2','gamma3','gamma4','delta','alpha','beta','runtime_seconds','status'])
        w.writerow(['P',*values(p_final),psec,'OK'])
        w.writerow(['IND',*values(i_final),isec,'OK'])
    print(f'PYTHON_CASE_P_SECONDS={psec:.17g}\nPYTHON_CASE_IND_SECONDS={isec:.17g}\nPYTHON_CASE_STATUS=PASS')
if __name__=='__main__':main()
