# C++/CUDA reproduction of Python Figure 1

## Reference experiment

The executable specification is [`FIGURE1_REFERENCE.md`](FIGURE1_REFERENCE.md).  It is derived from `scripts/run_figures_1_3_local.py`, `estimation_syn.py`, and `estimation_ind_syn.py`.  Figure 1 has nine settings: nu in {0.5,1.5,2.5}, with beta respectively (.033,.1,.234), (.037,.11,.256), and (.038,.113,.264).  There are 100 replications per setting, 10 workers, 100 observations per worker, 100 knots, five regressors, and 50 fixed synchronous rounds.  The true values are gamma=(-1,2,3,-2,1)^T, alpha=1, delta=1/2^2=.25, and theta=(alpha,beta).

Python generates one *global* 1,000-point GP realization before splitting consecutive blocks of 100 observations among workers.  Coordinates and knots are jittered grids; a full covariance is Cholesky-factored, `y=L eps` is sampled, `X` is standard normal, and `z=X gamma+y+epsilon`.  The production C++ program (`apps/sync_gpu_figure1_production.cu`) likewise creates one global coordinate/design/response set and then makes ten consecutive worker blocks.  Its `mt19937_64` stream is not PyTorch-identical, so fresh production agreement is distributional; deterministic shared fixtures are used for row-level parity.

With d=||s-t||/beta, the covariance is alpha exp(-d), alpha(1+sqrt(3)d)exp(-sqrt(3)d), or alpha(1+sqrt(5)d+5d^2/3)exp(-sqrt(5)d) for nu=.5,1.5,2.5.  Thus C=alpha R(beta), C_alpha=R, C_alpha_alpha=0, and C_alpha,beta=R_beta.  C++ calls the Python beta parameter `range`.

## Synchronous P update

For K=C(S*,S*), B=C(S,S*)K^-1, and M=delta(C(S,S)-BKB^T)+I, each round is implemented by `run_p_synchronous_round` in `src/p_sync_round.cu`:

1. `compute_mu_sigma_local` forms worker summaries.  Ordered worker means give Sigma=(K^-1+J delta mean_sigma)^-1 and mu=-Sigma(J delta mean_mu).
2. `compute_gamma_local` produces the local gamma and XX summaries; their ordered means give gamma=XX^-1 gamma_summary.
3. `compute_delta_theta_local` holds mu, Sigma, and gamma fixed, forms local P objective derivatives in parameter order (delta,alpha,range), and returns a 3-vector/3x3 Hessian.  The server mean is multiplied by J, receives the global K correction, then applies absolute-eigenvalue flooring at .01 and positivity-only step halving.  There is no convergence stopping: Figure 1 always uses 50 rounds.

The local P scalar is `-n log(delta)-log|local_inv|+delta(r^T local_inv r+tr(local_inv B Sigma B^T))`, with r=X gamma-z+B mu.  Worker reductions are explicitly in vector order, not nondeterministic GPU reductions.

## IND

`run_ind_synchronous_round` in `src/ind.cu` first aggregates X^T V^-1 z and X^T V^-1 X for gamma.  It then averages the local IND objective derivatives for V=C+delta^-1 I, performs the same modified-Hessian Newton update and positivity halving, and runs 50 fixed rounds.  `gpu_solve` uses cuSOLVER POTRF and cuBLAS TRSM for IND local solves; small aggregation, Hessian algebra, and Newton logic remain host code.

## Current GPU/CPU boundary

CUDA paths include `matern_with_derivatives` (`src/covariance.cu`), covariance kernels, GPU POTRF/TRSM/GEMM in `compute_mu_sigma_local`, `compute_gamma_local`, and IND.  The fused FP64 `cnk_fused` kernel produces alpha, range, alpha-alpha, alpha-range, and range-range Cnk derivatives; it is active through `make_cnk_derivative_workspace`.  CPU code still performs server reductions, 3x3 eigendecomposition/Newton logic, much derivative matrix algebra, and the active B-second right solves: `rs(a,K)=transpose(solve_spd(K,transpose(a)))` in `src/b_second_derivatives.cu`.

`PDerivativeWorkspace` owns B/M/A derivative products for one worker fixed-state evaluation. `PGlobalDerivativeWorkspace` owns K and its first/second derivatives for one joint-update round. `CnkDerivativeWorkspace` owns the five worker-cross-covariance derivative matrices. Their intended dependency is global K state plus worker Cnk state, then B, M, A, local derivatives, ordered reduction, and Newton.

The standalone `gpu_right_spd_solve` helper in `src/gpu_context.cu` is **not active in the estimator**. It converts host row-major X and L to column-major buffers and applies two FP64 DTRSM calls to solve K Y^T=X^T. CPU-vs-GPU parity was 6.39488e-14 with residual 6.03961e-14.

## Validation and performance

Current P iteration-1 parity is 5.7454e-14; latest 50-round trajectory maximum is 1.3322676295501878e-14 (iteration 16, alpha). Fused Cnk output matched `matern_with_derivatives` exactly for all three nu values on the fixture. Existing files under `results/figure1_cpp_gpu_faithful/parity` additionally record gamma, mu/Sigma, local derivative, Hessian, and IND checks.

P timing evolved from roughly 926 seconds before the fixed-state cache to 85.681 seconds for the measured baseline, 52.753 seconds after the derivative workspace, 52.004 seconds after global K reuse, and 51.542 seconds after fused Cnk integration. The active bottleneck is B-second: 24.584 seconds of 51.542 seconds. One round measured 320 CPU right-side solves (32 per worker), consuming .483431 of .514020 seconds of B-second time. This is the next optimization target; no GPU right-side solve has yet been integrated.
