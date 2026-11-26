#!/usr/bin/env python3
"""Dense CPU proof for the fixed-state synchronous-P H(delta, theta) slice.

This deliberately does not call any GPU code.  It uses the parity-clean
identities below, and validates every scalar summand with Torch AD and a
central difference.  theta is (alpha, range), delta is physical (not logged).
"""
import torch

torch.set_default_dtype(torch.float64)


def cov(x, y, theta):
    alpha, rho = theta
    distance = torch.cdist(x, y)
    s = torch.sqrt(torch.tensor(5.0)) * distance / rho
    return alpha * (1 + s + s * s / 3) * torch.exp(-s)


def cov_first(x, y, theta, which):
    alpha, rho = theta
    distance = torch.cdist(x, y)
    s = torch.sqrt(torch.tensor(5.0)) * distance / rho
    base = (1 + s + s * s / 3) * torch.exp(-s)
    if which == 0:
        return base
    return alpha * torch.exp(-s) * (s * s + s * s * s) / (3 * rho)


def pieces(v, which):
    d, alpha, rho = v
    theta = torch.stack((alpha, rho))
    K = cov(KNOTS, KNOTS, theta)
    C = cov(SITES, KNOTS, theta)
    Cjj = cov(SITES, SITES, theta)
    B = torch.linalg.solve(K, C.T).T
    ci, ki, cjji = (cov_first(SITES, KNOTS, theta, which),
                    cov_first(KNOTS, KNOTS, theta, which),
                    cov_first(SITES, SITES, theta, which))
    Bi = torch.linalg.solve(K, ci.T).T - B @ torch.linalg.solve(K, ki.T).T
    A = Cjj - B @ K @ B.T
    # Differentiate Cjj - B K B' explicitly.  This is independent of the
    # algebraically shorter Cjj - B C' form and catches B/K layout mistakes.
    Ai = cjji - Bi @ K @ B.T - B @ ki @ B.T - B @ K @ Bi.T
    M = torch.eye(3) + d * A
    W = torch.linalg.inv(M)
    r = X @ GAMMA - Z + B @ MU
    ri = Bi @ MU
    Q = B @ SIGMA @ B.T
    Qi = Bi @ SIGMA @ B.T + B @ SIGMA @ Bi.T
    T = d * Ai
    u = W @ r
    return d, W, r, u, B, Bi, A, Ai, T, ri, Q, Qi


def objective(v):
    d, W, r, u, B, Bi, A, Ai, T, ri, Q, Qi = pieces(v, 0)
    # Rebuild without selecting a theta derivative.
    theta = v[1:]
    K = cov(KNOTS, KNOTS, theta); C = cov(SITES, KNOTS, theta)
    B = torch.linalg.solve(K, C.T).T
    A = cov(SITES, SITES, theta) - B @ K @ B.T
    M = torch.eye(3) + v[0] * A
    W = torch.linalg.inv(M); r = X @ GAMMA - Z + B @ MU; Q = B @ SIGMA @ B.T
    return -3 * torch.log(v[0]) + torch.logdet(M) + v[0] * (r @ W @ r + torch.trace(W @ Q))


def delta_terms(v, which):
    d, W, r, u, B, Bi, A, Ai, T, ri, Q, Qi = pieces(v, which)
    W2 = W @ W
    return torch.stack((
        torch.trace(W @ T @ W) / d,
        2 * (ri @ W @ u),
        -2 * (u @ W @ T @ u),
        2 * torch.trace(Bi @ SIGMA @ B.T @ W2),
        -2 * torch.trace(B @ SIGMA @ B.T @ W @ T @ W2),
    ))


def delta_gradient(v):
    d, W, r, u, B, Bi, A, Ai, T, ri, Q, Qi = pieces(v, 0)
    return -torch.trace(W) / d + u @ u + torch.trace(W @ Q @ W)


def delta_gradient_sources(v, which):
    """AD-isolatable sources of the five delta-route terms.

    detach is used only to hold the *other* factor fixed, exactly matching
    the product-rule split; it is not used in the total objective check.
    """
    d, W, r, u, B, Bi, A, Ai, T, ri, Q, Qi = pieces(v, which)
    # Reconstruct r/Q paths without a theta-dependent inverse, and inverse
    # paths without a theta-dependent r/Q, respectively.
    r_path = X @ GAMMA - Z + B @ MU
    q_path = B @ SIGMA @ B.T
    u_r = W.detach() @ r_path
    u_w = W @ r_path.detach()
    return (
        -torch.trace(W) / d,
        u_r @ u_r,
        u_w @ u_w,
        torch.trace(W.detach() @ q_path @ W.detach()),
        torch.trace(W @ q_path.detach() @ W),
    )


def theta_gradient(v, which):
    d, W, r, u, B, Bi, A, Ai, T, ri, Q, Qi = pieces(v, which)
    return torch.trace(W @ T) + d * (2 * ri @ u - u @ T @ u + torch.trace(W @ Qi) - torch.trace(W @ T @ W @ Q))


def fd_scalar(f, v, axis, eps=2e-5):
    h = eps * max(1., abs(float(v[axis])))
    vp, vm = v.detach().clone(), v.detach().clone()
    vp[axis] += h; vm[axis] -= h
    return (f(vp) - f(vm)) / (2 * h)


def check(label, f, v, axis, analytic, ffd=None):
    vv = v.detach().clone().requires_grad_(True)
    ad = torch.autograd.grad(f(vv), vv)[0][axis]
    fd = fd_scalar(ffd or f, v, axis)
    ae, fe = abs(float(analytic - ad)), abs(float(analytic - fd))
    print(f"{label} ANALYTIC={float(analytic):+.16e} AD_ABS={ae:.3e} FD_ABS={fe:.3e}")
    return max(ae, fe)


def delta_gradient_sources_fd(v0, which):
    """Finite-difference counterparts with the opposite product factor fixed."""
    d0, W0, r0, u0, B0, Bi0, A0, Ai0, T0, ri0, Q0, Qi0 = pieces(v0, which)
    W0, r0, Q0 = W0.detach(), r0.detach(), Q0.detach()
    def source(k):
        def f(v):
            d, W, r, u, B, Bi, A, Ai, T, ri, Q, Qi = pieces(v, which)
            if k == 0: return -torch.trace(W) / d
            if k == 1: return r @ W0 @ W0 @ r
            if k == 2: return r0 @ W @ W @ r0
            if k == 3: return torch.trace(W0 @ Q @ W0)
            return torch.trace(W @ Q0 @ W)
        return f
    return source


SITES = torch.tensor([[0., 0.], [1., .2], [.3, 1.1]])
KNOTS = torch.tensor([[0., 0.], [1.5, 1.5]])
Z = torch.tensor([1.2, -.4, .7]); X = torch.ones((3, 1)); GAMMA = torch.tensor([.15])
MU = torch.tensor([.2, -.1]); SIGMA = torch.tensor([[.7, .05], [.05, .4]])


def main():
    v = torch.tensor([1.7, 1.2, .8])
    worst = 0.
    # Route A: differentiate the already parity-clean delta gradient.
    for q, name in enumerate(("ALPHA", "RANGE")):
        terms = delta_terms(v, q)
        source = delta_gradient_sources(v, q)
        source_fd = delta_gradient_sources_fd(v, q)
        for k in range(5):
            worst = max(worst, check(f"DELTA_ROUTE_{name}_TERM{k}",
                                     lambda x, k=k, q=q: delta_gradient_sources(x, q)[k],
                                     v, q + 1, terms[k], source_fd(k)))
        total = terms.sum()
        worst = max(worst, check(f"DELTA_ROUTE_{name}_TOTAL", delta_gradient, v, q + 1, total))
        # Route B: independently differentiate the clean theta gradient in delta.
        # Re-evaluate with one leaf: the expression is kept separate from route A.
        leaf = v.detach().clone().requires_grad_(True)
        b_value = torch.autograd.grad(theta_gradient(leaf, q), leaf)[0][0]
        fd_b = fd_scalar(lambda x, q=q: theta_gradient(x, q), v, 0)
        b_err, fd_err = abs(float(total - b_value)), abs(float(total - fd_b))
        print(f"THETA_ROUTE_{name}_TOTAL ANALYTIC={float(total):+.16e} AD_ABS={b_err:.3e} FD_ABS={fd_err:.3e}")
        worst = max(worst, b_err, fd_err)
        h = torch.autograd.functional.hessian(objective, v)
        parity = abs(float(total - h[0, q + 1])); symmetry = abs(float(h[0, q + 1] - h[q + 1, 0]))
        print(f"AUTOGRAD_H_{name}_ABS={parity:.3e} SYMMETRY_ABS={symmetry:.3e}")
        worst = max(worst, parity, symmetry)
    print(f"DENSE_CPU_MIXED_HESSIAN_MAX_ABS={worst:.3e}")
    print("DENSE_CPU_MIXED_HESSIAN=" + ("PASS" if worst <= 2e-8 else "FAIL"))
    return 0 if worst <= 2e-8 else 1


if __name__ == "__main__":
    raise SystemExit(main())
