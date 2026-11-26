#!/usr/bin/env python3
"""One exact, CPU-only P update on an exported two-worker Figure-1 fixture."""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'new_bersion_of_python_code'/'Asyn_Fed_Spatial'))
from src.kernel import exponential_kernel, onedif_kernel, Matern_2_5_kernel
from src.estimation_syn import GlobalComputation, LocalComputation, Parameter

def array(x): return x.detach().cpu().numpy().astype(np.float64).tolist()
def main():
  ap=argparse.ArgumentParser(); ap.add_argument('--fixture-dir',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--timing-json',type=Path); ap.add_argument('--threads',type=int,default=40); a=ap.parse_args()
  os.environ['CUDA_VISIBLE_DEVICES']=''; torch.set_num_threads(a.threads); torch.set_default_dtype(torch.float64)
  meta=json.loads((a.fixture_dir/'metadata.json').read_text()); nu=float(meta['nu']); kernel_base={.5:exponential_kernel,1.5:onedif_kernel,2.5:Matern_2_5_kernel}[nu]
  theta=torch.tensor([[1.],[float(meta['beta'])]],dtype=torch.float64)
  def kernel(x,y=None,theta_arg=None): return kernel_base(x,y,theta=theta if theta_arg is None else theta_arg)
  sites=torch.from_numpy(np.load(a.fixture_dir/'sites.npy')).to(torch.float64); z=torch.from_numpy(np.load(a.fixture_dir/'z.npy')).reshape(-1,1).to(torch.float64); x=torch.from_numpy(np.load(a.fixture_dir/'X.npy')).to(torch.float64); knots=torch.from_numpy(np.load(a.fixture_dir/'knots.npy')).to(torch.float64)
  assert sites.shape[0]==4000 and knots.shape[0]==400 and x.shape[1]==5
  data=torch.hstack((sites,z,x)); chunks=[data[:2000],data[2000:]]
  local=[LocalComputation(v,kernel,knots) for v in chunks]
  for v in local: v.type_LR='P'
  p=Parameter(None,None,torch.tensor([[-1.],[2.],[3.],[-2.],[1.]],dtype=torch.float64),torch.tensor(.25,dtype=torch.float64),theta)
  # Exact Server ordering: average compact stats, global mu/Sigma; then gamma;
  # then local 3-vector objective/gradient/Hessian and global K correction.
  p.index=(0,None,0)
  measured_start=time.perf_counter(); t=time.perf_counter(); ms=[v.compute_local_for_mu_sigma(p) for v in local]; local_mu_sigma_seconds=time.perf_counter()-t
  mean_mu=sum(v['mu'] for v in ms)/2; mean_sigma=sum(v['sigma'] for v in ms)/2
  global_c=GlobalComputation(knots,kernel,1.,type_LR='P'); global_c.set_global_para_ind_quantities({'n':torch.tensor(2000,dtype=torch.float64),'J':2}); t=time.perf_counter(); after_ms=global_c.update_mu_Sigma(p,{'mu':mean_mu,'sigma':mean_sigma}); global_mu_sigma_seconds=time.perf_counter()-t
  t=time.perf_counter(); gs=[v.compute_local_for_gamma(after_ms) for v in local]; local_gamma_seconds=time.perf_counter()-t
  mean_gamma=sum(v['gamma'] for v in gs)/2; mean_xx=sum(v['XX'] for v in gs)/2
  t=time.perf_counter(); after_gamma=global_c.update_gamma(after_ms,{'gamma':mean_gamma,'XX':mean_xx}); global_gamma_seconds=time.perf_counter()-t
  t=time.perf_counter(); dt=[v.compute_local_for_theta_delta(after_gamma) for v in local]; local_delta_theta_seconds=time.perf_counter()-t
  # The original method returns only derivative tensors.  Record its exact
  # fixed-state scalar objective too, so all 13 CUDA producer slots have a
  # Python-originated counterpart.
  objectives=[]
  for v in local:
    b,_=v.local_B(after_gamma.theta); inv=v.local_inv(after_gamma.theta,after_gamma.delta)
    err=v.local_errorv(after_gamma.gamma); residual=err+b@after_gamma.mu
    bsigma=b@after_gamma.sigma@b.T
    objectives.append((-v.n*torch.log(after_gamma.delta)-torch.logdet(inv)+after_gamma.delta*(residual.T@inv@residual+torch.trace(inv@bsigma))).squeeze())
  mean_grad=sum(v['grad'] for v in dt)/2; mean_hessian=sum(v['hessian'] for v in dt)/2
  t=time.perf_counter(); correction=global_c.Grad_Hessian_f(after_gamma)
  grad=2*mean_grad.reshape(-1,1)+torch.cat((torch.zeros((1,1),dtype=torch.float64),correction['grad']),0)
  correction_hessian=torch.cat((torch.zeros((1,2),dtype=torch.float64),correction['hessian']),0)
  correction_hessian=torch.cat((torch.zeros((3,1),dtype=torch.float64),correction_hessian),1)
  hess=2*mean_hessian+correction_hessian
  eigval,eigvec=torch.linalg.eigh(hess); mod=eigvec@torch.diag(torch.maximum(eigval.abs(),torch.tensor(.01)))@eigvec.T; direction=torch.linalg.inv(mod)@grad
  after_dt=global_c.update_delta_theta(after_gamma,{'grad':mean_grad,'hessian':mean_hessian}); global_delta_theta_seconds=time.perf_counter()-t; measured_seconds=time.perf_counter()-measured_start
  output={'fixture_metadata':meta,'workers':2,'local_mu_sigma':[{'mu':array(v['mu']),'sigma':array(v['sigma'])} for v in ms], 'mean_mu':array(mean_mu),'mean_sigma':array(mean_sigma),'after_mu_sigma':{'mu':array(after_ms.mu),'sigma':array(after_ms.sigma)},'local_gamma':[{'gamma':array(v['gamma']),'xx':array(v['XX'])} for v in gs], 'mean_gamma':array(mean_gamma),'mean_xx':array(mean_xx),'after_gamma':array(after_gamma.gamma),'local_delta_theta':[{'objective':float(objectives[i]),'gradient':array(v['grad']),'hessian':array(v['hessian'])} for i,v in enumerate(dt)], 'mean_gradient':array(mean_grad),'mean_hessian':array(mean_hessian),'global_correction':{'gradient':array(correction['grad']),'hessian':array(correction['hessian'])},'global_gradient':array(grad),'global_hessian':array(hess),'newton_direction':array(direction),'next':{'delta':array(after_dt.delta),'theta':array(after_dt.theta)}}
  a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(output,indent=2)); print('PYTHON_M400_REFERENCE=PASS')
  if a.timing_json:
    a.timing_json.parent.mkdir(parents=True,exist_ok=True)
    a.timing_json.write_text(json.dumps({'warmup_rounds':0,'fixture_setup_excluded':True,'threads':a.threads,'local_mu_sigma_seconds':local_mu_sigma_seconds,'global_mu_sigma_seconds':global_mu_sigma_seconds,'local_gamma_seconds':local_gamma_seconds,'global_gamma_seconds':global_gamma_seconds,'local_delta_theta_seconds':local_delta_theta_seconds,'global_delta_theta_seconds':global_delta_theta_seconds,'measured_round_seconds':measured_seconds},indent=2))
if __name__=='__main__': main()
