# Figure 1 authoritative Python reference

This document records the executable specification for the C++/CUDA port.  It
is derived from `scripts/run_figures_1_3_local.py` and the original classes in
`new_bersion_of_python_code/Asyn_Fed_Spatial/src/`; it does not describe the
older non-comparable `results/cpp_gpu_figure1_from_scratch` prototype.

## Experiment matrix

Figure 1 has nine configurations, all with 100 replications (`replicate=0..99`,
Python seed `1 + replicate`), 10 workers, 100 observations per worker, 100
knots, direct initialisation, step size 1, 50 fixed synchronous iterations,
and the P approximation for both estimators.

| Matérn nu | length scales beta |
| --- | --- |
| .5 | .033, .1, .234 |
| 1.5 | .037, .11, .256 |
| 2.5 | .038, .113, .264 |

The common model values are `theta=[alpha,beta]=[1,beta]`, five regression
coefficients `gamma=[-1,2,3,-2,1]^T`, noise standard deviation 2, and hence
noise precision `delta=1/2^2=.25`.

## Data generation

`GPPSampleGeneratorUnitSquare.generate_obs_gp(m=100, method="grid")` is used.
For `n=1000`, it makes a 32 by 32 regular coordinate grid, adds independent
uniform jitter in `[-.4,.4]` in each dimension, transforms coordinates by
`(grid-.5+jitter)/32`, and takes 1000 points from the 1024 candidates.  A
separate 10 by 10 jittered grid is used for the 100 knots by the same formula.

The generator draws one global GP vector `y=L eps`, where `L` is the Cholesky
factor of the 1000 by 1000 Matérn covariance and `eps` is standard normal.  It
draws a 1000 by 5 standard-normal design matrix `X` and iid N(0,2^2) noise,
then writes `z=X gamma + y + epsilon`.  The resulting *single* 1000-row array
`[s_x,s_y,z,X_1,...,X_5]` is partitioned by `torch.chunk(data,10,dim=0)` into
ten consecutive 100-row worker arrays.  A C++ generator must preserve this
one-global-dataset then consecutive-chunk design (though its RNG stream need
not be PyTorch-identical).

For distances `d=||s-t||/beta`, Python's covariance is:

* nu=.5: `alpha exp(-d)`;
* nu=1.5: `alpha (1+sqrt(3)d) exp(-sqrt(3)d)`;
* nu=2.5: `alpha (1+sqrt(5)d+5d^2/3) exp(-sqrt(5)d)`.

## IND (`estimation_ind_syn.py`, P)

At direct initialisation `(gamma,delta,theta)` is exactly the common starting
parameter.  At each of 50 rounds, workers first form
`V_j=C_j(theta)+delta^-1 I`, then report
`XX_j=X_j^T V_j^-1 X_j` and `g_j=X_j^T V_j^-1 z_j`; the server averages those
summaries and sets `gamma=(mean XX_j)^-1 mean g_j`.

Workers next evaluate the P local objective
`[log|V_j|+(X_j gamma-z_j)^T V_j^-1(X_j gamma-z_j)]/n_j` and its gradient and
Hessian with respect to `[delta,alpha,beta]`.  The server averages both,
eigen-decomposes the Hessian, replaces every eigenvalue by
`max(abs(eigenvalue),.01)`, takes the Newton step with step size 1, and halves
that step until delta, alpha, and beta are all positive.  Its result begins
the next gamma round.  There is no convergence stopping condition: exactly 50
rounds are run and the last pre-update record in `Server.params` is plotted.

### Joint-update conventions (authoritative)

For both IND and low-rank P, the local autodiff variable is exactly
`var=torch.cat([delta.reshape(1), theta.reshape(-1)])`; therefore the
parameter, gradient, and Hessian order is `(delta, alpha, beta)`.  Each local
function returns derivatives of the displayed scalar objective.  IND divides
its scalar by local `n`; low-rank P does not.

Workers are reduced in worker-ID order and the server divides every summary by
`J`. IND uses this mean directly. Low-rank P multiplies its mean local
gradient/Hessian by `J`, then adds the zero-padded derivatives of
`mu^T K^-1 mu + tr(K^-1 Sigma) + log|K|`.  The direction is
`modified_hessian^-1 @ gradient`; the update is `old - step_size*direction`
starting at step size 1. `torch.linalg.eigh` is used, every eigenvalue becomes
`max(abs(eigenvalue),.01)`, and a non-positive delta/alpha/beta causes repeated
step halving. There is no clipping, pseudoinverse, convergence stop, or line
search.

## Low-rank (`estimation_syn.py`, P, `dl_th_together=True`)

Let `K=C(S*,S*)`, `B_j=C(S_j,S*)K^-1`, and
`R_j=C(S_j,S_j)-B_j K B_j^T + delta^-1 I`.  The code stores
`local_inv=[delta(C-BKB^T)+I]^-1=delta^-1 R_j^-1`.

For each of 50 rounds, the exact order is:

1. **mu/Sigma:** workers send `B_j^T local_inv B_j` and
   `B_j^T local_inv(X_j gamma-z_j)`; server averages, then uses
   `Sigma=(K^-1+J delta mean_sigma)^-1` and
   `mu=-Sigma(J delta mean_mu)`.
2. **gamma:** workers send
   `X_j^T local_inv(z_j-B_j mu)` and `X_j^T local_inv X_j`; server averages
   and sets `gamma=(mean_XX)^-1 mean_gamma`.
3. **joint delta/theta:** each worker evaluates the Python P local scalar
   `-n log(delta)-log|local_inv| + delta*(r^T local_inv r +
   tr(local_inv B Sigma B^T))`, where `r=X gamma-z+B mu`, and reports
   gradient/Hessian for `[delta,alpha,beta]`.  The server averages these
   summaries, multiplies both by `J`, then adds the zero-padded derivative and
   Hessian of `mu^T K^-1 mu + tr(K^-1 Sigma) + log|K|` (the first delta row and
   column are zero).  It applies the same absolute-eigenvalue .01 floor and
   positive-parameter step halving.  The updated delta/theta begin the next
   mu/Sigma block.

The port must preserve worker-ID ordered reductions and compare every local and
server summary against a deterministic shared fixture before production.

### Exact P scalar differentiated in the joint block

During `compute_local_for_theta_delta`, the current `mu`, `Sigma`, and
`gamma` are fixed constants.  With `C=C(S_j,S_j;alpha,beta)`,
`K=C(S*,S*;alpha,beta)`, `B=C(S_j,S*;alpha,beta)K^-1`, and
`A=C-B K B^T`, Python defines `inv=(delta*A+I)^-1`,
`r=X_j gamma-z_j+B mu`, and `Q=B Sigma B^T`.  Its exact P scalar is

`f_j(delta,alpha,beta) = -n_j log(delta) - logdet(inv)
                              + delta*(r^T inv r + trace(inv Q))`.

There is no factor one-half and `f_j` is not divided by `n_j`.  Autograd
returns its gradient and Hessian in `(delta,alpha,beta)` order.  Delta enters
both explicit terms and `inv`; alpha/beta enter C, K, B, A, r, and Q.  The
server averages worker derivatives, multiplies them by `J`, and adds the
zero-padded derivatives of the fixed-state global term
`g(alpha,beta)=mu^T K^-1 mu + trace(K^-1 Sigma)+logdet(K)` before applying the
modified-Hessian Newton rule documented above.

## Plot definition

`plot_python_figures_1_3_from_scratch.py:plot1` makes a 3 by 6 panel figure.
For each nu row and beta column, column `2*c` is a sigma/alpha boxplot and
column `2*c+1` is a beta boxplot.  Each has groups in the order `Ind`,
`Low-rank`, a green triangle for mean, white box fill, grid, and a red dashed
horizontal true-value line.  The plotted data are the final `theta[0]` and
`theta[1]` for all 100 replications.
