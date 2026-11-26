# Exact reduced-space audit for synchronous low-rank P

## Scope and notation

For one worker, the Python P path and the CUDA path both form

\[
 S(\theta)=C_\theta-B_\theta K_\theta B_\theta^T,\qquad
 M(\delta,\theta)=I_n+\delta S(\theta).
\]

`C_theta` is the dense Matérn covariance of the worker sites, `B` is
`n x m`, and `K` is `m x m`.  The implementation is directly visible in
`estimation_syn.py:local_inv` and `gpu_context.cu:make_m`: both construct
`I + delta * (C - B K B^T)`.  The conditional covariance is not diagonal,
not low rank, and is generally full rank for distinct sites.

The fixed-state local P objective used by the delta/theta producer can be
written (up to a parameter-independent constant) as

\[
 f=-n\log\delta+\log|M|+\delta\{r^TWr+\operatorname{tr}(WQ)\},
 \quad W=M^{-1},\quad r=e+B\mu,\quad Q=B\Sigma B^T.
\]

This is the same dense `inv` objective in Python `f_scalar`.

## Formal Woodbury attempt

Let \(H=I+\delta C\) and \(U=\sqrt{\delta}B\).  The exact determinant
lemma and Woodbury identities are

\[
 M=H-UKU^T,
\]
\[
 \log|M|=\log|H|+\log|K|+
 \log|K^{-1}-U^TH^{-1}U|,
\]
\[
 W=H^{-1}+H^{-1}U(K^{-1}-U^TH^{-1}U)^{-1}U^TH^{-1}.
\]

Consequently `B'WB`, `W r`, and every low-rank-Q contraction can be
expressed using `H^{-1}B`, `H^{-1}r`, and `m x m` matrices.  Their sizes
are:

| quantity | dimensions | once H is factored |
|---|---:|---:|
| `H` / Cholesky(H) | n x n | dense factor required |
| `H^{-1}B` | n x m | two matrix-RHS solves |
| `H^{-1}r` | n | two solves |
| `G=K^{-1}-U'H^{-1}U` | m x m | POTRF/solve |

But factorizing dense `H` costs \(O(n^3)\), and producing
`H^{-1}B` costs \(O(n^2m)\).  Neither operation can be reduced to
\(O(nm^2+m^3)\) without an additional representation of `C_theta`.

## Derivative mapping and obstruction

For parameters \(p,q\in\{\delta,\alpha,\rho\}\), define
\(A_p=M_p\), \(A_{pq}=M_{pq}\), `r_p=B_p mu`, and
`Q_p=B_p Sigma B' + B Sigma B_p'` (with the analogous second derivative).

\[
 A_\delta=S,
\]
\[
 A_t=\delta[C_t-B_tKB^T-BK_tB^T-BKB_t^T],
\]
\[
 A_{tu}=\delta[C_{tu}-(BKB^T)_{tu}],\quad t,u\in\{\alpha,\rho\}.
\]

The generic exact identities used by an autodiff/Hessian implementation are

\[
 W_p=-WA_pW,
\]
\[
 (\log|M|)_p=\operatorname{tr}(WA_p),
\]
\[
 (\log|M|)_{pq}=\operatorname{tr}(WA_{pq})-
 \operatorname{tr}(WA_qWA_p).
\]

Differentiating `delta*(r'Wr + tr(WQ))` adds only vector products and traces
of the same form.  The low-rank pieces of `A_t`, `A_tu`, `Q_t`, and `Q_tu`
reduce through `B`, `B_t`, `B_tu`, and `m x m` Gram matrices.  However,
`C_t` and `C_tu` are dense Matérn derivative matrices.  Terms such as

\[
 \operatorname{tr}(W C_t),\quad
 \operatorname{tr}(W C_{tu}),\quad
 \operatorname{tr}(W C_u W C_t),\quad
 r^T W C_t W r
\]

remain full-rank dense-covariance operations.  There is no exact cyclic-trace
rearrangement that turns an arbitrary dense `C_t` into an `m x m` product.

## Complexity conclusion

An exact P implementation with the current Python parameterization has a
lower-bound dense covariance factor/solve dependency:

\[
 O(n^3+n^2m)\quad\text{per changed theta state},
\]

before any `m x m` low-rank correction.  Therefore a
`FEDSPATIAL_REDUCED_DELTA_THETA=1` path that promises exact
`O(nm^2+m^3)` would be mathematically invalid unless it changes the model.

Valid future choices are:

1. Change P to an explicitly diagonal/sparse/structured base covariance and
   update the Python reference accordingly; then Woodbury gives the requested
   reduced-space formulas.
2. Keep exact P and optimize the dense base factor/action (hierarchical,
   sparse precision, or a mathematically approved approximation), with a new
   parity reference and experiment label.
3. Retain the current exact P semantics and use the verified batching only;
   this cannot make the 15k dense P theta Hessian timing-ready.

No reduced branch is enabled from this note because doing so would violate
the required Python parity.
