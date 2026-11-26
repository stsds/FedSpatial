#!/usr/bin/env python3
"""Export one Figure-1 Python-generated global dataset without estimator output."""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "new_bersion_of_python_code" / "Asyn_Fed_Spatial"))
from src.data_generation import GPPSampleGeneratorUnitSquare
from src.kernel import exponential_kernel, onedif_kernel, Matern_2_5_kernel

SCENARIOS = [(0.5,.033),(0.5,.1),(0.5,.234),(1.5,.037),(1.5,.11),(1.5,.256),(2.5,.038),(2.5,.113),(2.5,.264)]
GAMMA = [-1,2,3,-2,1]

def main():
    a=argparse.ArgumentParser(); a.add_argument('--scenario-id',type=int,required=True); a.add_argument('--replication',type=int,required=True); a.add_argument('--output-root',type=Path,required=True); a.add_argument('--n',type=int,default=1000); a.add_argument('--knots',type=int,default=100); a.add_argument('--workers',type=int,default=10); a.add_argument('--two-worker-15k-target',action='store_true'); a.add_argument('--two-worker-2k-timing-target',action='store_true'); q=a.parse_args()
    if q.two_worker_15k_target:
        if (q.scenario_id, q.replication, q.n, q.knots, q.workers) != (4, 0, 30000, 2000, 2):
            raise ValueError('--two-worker-15k-target requires scenario 4, replication 0, n=30000, knots=2000, workers=2')
    if q.two_worker_2k_timing_target:
        if (q.scenario_id, q.replication, q.n, q.knots, q.workers) != (4, 0, 4000, 2000, 2):
            raise ValueError('--two-worker-2k-timing-target requires scenario 4, replication 0, n=4000, knots=2000, workers=2')
    if q.n <= 0 or q.knots <= 0 or q.workers <= 0 or q.n % q.workers:
        raise ValueError('--n must be positive and divisible by --workers; --knots/--workers positive')
    nu,beta=SCENARIOS[q.scenario_id]; seed=1+q.replication; alpha=1.; noise_level=2.; delta=.25
    kernel_fn={.5:exponential_kernel,1.5:onedif_kernel,2.5:Matern_2_5_kernel}[nu]
    theta=torch.tensor([alpha,beta],dtype=torch.float64); kernel=lambda x,y=None:kernel_fn(x,y,theta=theta)
    # Export through the original Figure-1 generator.  The optional component
    # return is observational only: it makes already-generated y/epsilon/X
    # available and does not consume any additional random numbers.
    torch.manual_seed(seed)
    sampler=GPPSampleGeneratorUnitSquare(q.n,kernel,GAMMA,noise_level,seed=seed)
    (data, knots, components), = sampler.generate_obs_gp(q.knots, "grid", return_components=True)
    sites=data[:, :2]
    X=components['X']
    latent=components['latent']
    noise=components['noise']
    z=data[:, 2]
    out=q.output_root/f"scenario_{q.scenario_id:02d}"/f"rep_{q.replication:03d}"; out.mkdir(parents=True,exist_ok=True)
    arrays={'sites':sites,'knots':knots,'X':X,'latent':latent,'noise':noise,'z':z}
    for name,t in arrays.items(): np.save(out/f'{name}.npy',t.numpy())
    # Preserve the generator's grouping: it forms X@gamma + epsilon before
    # adding the latent process.  Re-associating those FP64 additions is not
    # bitwise equivalent, even though it is algebraically the same.
    gamma_column=torch.tensor(GAMMA,dtype=torch.float64).reshape(-1,1)
    expected_z=latent.reshape(-1,1)+(X @ gamma_column+noise.reshape(-1,1))
    if not torch.equal(z.reshape(-1,1), expected_z):
        raise RuntimeError('response identity failed')
    hashes={name:hashlib.sha256(np.load(out/f'{name}.npy').tobytes()).hexdigest() for name in arrays}
    (out/'metadata.json').write_text(json.dumps({'scenario_id':q.scenario_id,'replication':q.replication,'seed':seed,'nu':nu,'alpha':alpha,'beta':beta,'delta':delta,'n':q.n,'p':5,'knots':q.knots,'workers':q.workers,'observations_per_worker':q.n//q.workers,'partition_order':'global_then_consecutive','generator':'original GPPSampleGeneratorUnitSquare.generate_obs_gp','response_identity':'z = latent + X@gamma + noise','array_sha256':hashes},indent=2))
    for name,t in arrays.items():
        x=np.load(out/f'{name}.npy'); assert x.dtype==np.float64 and np.isfinite(x).all() and np.array_equal(x,t.numpy())
    print(out)
if __name__=='__main__': main()
