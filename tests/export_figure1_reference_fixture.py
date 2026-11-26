#!/usr/bin/env python3
"""Export one deterministic *reference-only* Figure 1 fixture.

This script is not part of the production C++ experiment.  It provides the
identical FP64 data and Python trajectories needed to gate the native port.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / "new_bersion_of_python_code" / "Asyn_Fed_Spatial"
sys.path.insert(0, str(REF))

from src.data_generation import GPPSampleGeneratorUnitSquare
from src.estimation_ind_syn import (GlobalComputation as IndGlobal,
                                    LocalComputation as IndLocal,
                                    Parameter as IndParameter, Server as IndServer,
                                    Worker as IndWorker)
from src.estimation_syn import (GlobalComputation as LrGlobal,
                                LocalComputation as LrLocal,
                                Parameter as LrParameter, Server as LrServer,
                                Worker as LrWorker)
from src.kernel import onedif_kernel

torch.set_default_dtype(torch.float64)
OUT = ROOT / "sync_gpu" / "tests" / "fixtures" / "figure1_reference"


def write_matrix(path: Path, value: torch.Tensor) -> None:
    value = value.detach().cpu().reshape(value.shape[0], -1)
    with path.open("w") as handle:
        handle.write(f"{value.shape[0]} {value.shape[1]}\n")
        for item in value.flatten():
            handle.write(f"{item.item():.17g}\n")


def record_trajectory(path: Path, parameters: list) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["iteration", "gamma_0", "gamma_1", "gamma_2", "gamma_3", "gamma_4",
                         "delta", "alpha", "beta"])
        for iteration, parameter in enumerate(parameters):
            writer.writerow([iteration, *parameter.gamma.flatten().tolist(), parameter.delta.item(),
                             *parameter.theta.flatten().tolist()])


def main() -> None:
    # Representative Figure 1 setting: nu=1.5, beta=.11, replication=0.
    # The general C++ port must additionally pass this fixture for nu=.5/.2.5.
    seed, nu, beta, workers, n_local, m, rounds = 1, 1.5, .11, 10, 100, 100, 51
    coefficients = [-1, 2, 3, -2, 1]
    theta = torch.tensor([1., beta], dtype=torch.float64)
    gamma = torch.tensor(coefficients, dtype=torch.float64).reshape(-1, 1)
    delta = torch.tensor(.25, dtype=torch.float64)
    kernel = lambda x, y=None: onedif_kernel(x, y, theta=theta)

    torch.manual_seed(seed)
    generator = GPPSampleGeneratorUnitSquare(workers * n_local, kernel, coefficients,
                                               noise_level=2., seed=seed)
    (data, knots), = generator.generate_obs_gp(m=m, method="grid")
    chunks = generator.data_split(data, workers)
    OUT.mkdir(parents=True, exist_ok=True)
    write_matrix(OUT / "global_data.txt", data)
    write_matrix(OUT / "knots.txt", knots)
    for worker_id, chunk in enumerate(chunks):
        write_matrix(OUT / f"worker_{worker_id}.txt", chunk)

    # Deterministic p=5 block fixtures at the exact first low-rank round.
    # These are deliberately exported before the 50-round runs so native gamma
    # and mu/Sigma routines can be validated independently.
    p0 = LrParameter(**{"gamma": gamma, "delta": delta, "theta": theta.reshape(-1, 1)})
    p0.index = (0, None, 0)
    local_mu_sigma = []
    for worker_id, chunk in enumerate(chunks):
        local = LrLocal(chunk, onedif_kernel, knots)
        local.type_LR = "P"
        value = local.compute_local_for_mu_sigma(p0)
        local_mu_sigma.append(value)
        write_matrix(OUT / f"p_initial_worker_{worker_id}_mu.txt", value["mu"])
        write_matrix(OUT / f"p_initial_worker_{worker_id}_sigma.txt", value["sigma"])
    mean_mu = sum(v["mu"] for v in local_mu_sigma) / workers
    mean_sigma = sum(v["sigma"] for v in local_mu_sigma) / workers
    global_lr = LrGlobal(knots, onedif_kernel, 1., type_LR="P")
    global_lr.set_global_para_ind_quantities({"J": workers, "n": torch.tensor(float(n_local))})
    p_after_mu_sigma = global_lr.update_mu_Sigma(p0, {"mu": mean_mu, "sigma": mean_sigma})
    write_matrix(OUT / "p_initial_global_mu.txt", p_after_mu_sigma.mu)
    write_matrix(OUT / "p_initial_global_sigma.txt", p_after_mu_sigma.sigma)
    gamma_locals = []
    for worker_id, chunk in enumerate(chunks):
        local = LrLocal(chunk, onedif_kernel, knots)
        local.type_LR = "P"
        value = local.compute_local_for_gamma(p_after_mu_sigma)
        gamma_locals.append(value)
        write_matrix(OUT / f"p_initial_worker_{worker_id}_gamma.txt", value["gamma"])
        write_matrix(OUT / f"p_initial_worker_{worker_id}_xx.txt", value["XX"])
    mean_gamma = sum(v["gamma"] for v in gamma_locals) / workers
    mean_xx = sum(v["XX"] for v in gamma_locals) / workers
    gamma_after = global_lr.update_gamma(p_after_mu_sigma, {"gamma": mean_gamma, "XX": mean_xx})
    write_matrix(OUT / "p_initial_global_gamma.txt", mean_gamma)
    write_matrix(OUT / "p_initial_global_xx.txt", mean_xx)
    write_matrix(OUT / "p_initial_gamma_new.txt", gamma_after.gamma)
    # Exact joint-block autograd outputs and the server's first accepted Newton
    # update.  These are parity inputs only, never production inputs.
    joint_locals = []
    for worker_id, chunk in enumerate(chunks):
        local = LrLocal(chunk, onedif_kernel, knots)
        local.type_LR = "P"
        value = local.compute_local_for_theta_delta(gamma_after)
        joint_locals.append(value)
        # Scalar objective corresponding to the already exported autograd
        # gradient/Hessian.  This is reference data for native parity only.
        local_b, _ = local.local_B(gamma_after.theta)
        local_r = local.local_errorv(gamma_after.gamma) + local_b @ gamma_after.mu
        local_inv = local.local_inv(gamma_after.theta, gamma_after.delta)
        local_q = local_b @ gamma_after.sigma @ local_b.T
        local_objective = (-n_local * torch.log(gamma_after.delta)
                           - torch.logdet(local_inv)
                           + gamma_after.delta * (local_r.T @ local_inv @ local_r
                                                  + torch.trace(local_inv @ local_q))).squeeze()
        write_matrix(OUT / f"p_initial_worker_{worker_id}_objective.txt",
                     local_objective.reshape(1, 1))
        write_matrix(OUT / f"p_initial_worker_{worker_id}_gradient.txt", value["grad"].reshape(-1, 1))
        write_matrix(OUT / f"p_initial_worker_{worker_id}_hessian.txt", value["hessian"])
    mean_gradient = sum(v["grad"] for v in joint_locals) / workers
    mean_hessian = sum(v["hessian"] for v in joint_locals) / workers
    correction = global_lr.Grad_Hessian_f(gamma_after)
    correction_gradient = torch.cat((torch.zeros(1, 1), correction["grad"].reshape(-1, 1)), dim=0)
    correction_hessian = torch.zeros(3, 3, dtype=torch.float64)
    correction_hessian[1:, 1:] = correction["hessian"]
    raw_gradient = workers * mean_gradient.reshape(-1, 1) + correction_gradient
    raw_hessian = workers * mean_hessian + correction_hessian
    write_matrix(OUT / "p_initial_global_local_gradient.txt", mean_gradient.reshape(-1, 1))
    write_matrix(OUT / "p_initial_global_local_hessian.txt", mean_hessian)
    write_matrix(OUT / "p_initial_global_correction_gradient.txt", correction_gradient)
    write_matrix(OUT / "p_initial_global_correction_hessian.txt", correction_hessian)
    write_matrix(OUT / "p_initial_global_raw_gradient.txt", raw_gradient)
    write_matrix(OUT / "p_initial_global_raw_hessian.txt", raw_hessian)
    eigenvalues, eigenvectors = torch.linalg.eigh(raw_hessian)
    modified_eigenvalues = torch.maximum(eigenvalues.abs(), torch.tensor(.01, dtype=torch.float64))
    modified_hessian = eigenvectors @ torch.diag(modified_eigenvalues) @ eigenvectors.T
    newton_direction = torch.linalg.inv(modified_hessian) @ raw_gradient
    accepted_step = global_lr.step_size
    candidate = torch.cat((gamma_after.delta.reshape(1), gamma_after.theta.reshape(-1))) - accepted_step * newton_direction.reshape(-1)
    while (candidate <= 0).any():
        accepted_step = accepted_step / 2
        candidate = torch.cat((gamma_after.delta.reshape(1), gamma_after.theta.reshape(-1))) - accepted_step * newton_direction.reshape(-1)
    write_matrix(OUT / "p_initial_global_raw_eigenvalues.txt", eigenvalues.reshape(-1, 1))
    write_matrix(OUT / "p_initial_global_modified_eigenvalues.txt", modified_eigenvalues.reshape(-1, 1))
    write_matrix(OUT / "p_initial_global_modified_hessian.txt", modified_hessian)
    write_matrix(OUT / "p_initial_global_newton_direction.txt", newton_direction.reshape(-1, 1))
    write_matrix(OUT / "p_initial_global_accepted_step.txt", torch.tensor([[accepted_step]], dtype=torch.float64))
    updated_joint = global_lr.update_delta_theta(gamma_after, {"grad": mean_gradient, "hessian": mean_hessian})
    write_matrix(OUT / "p_initial_joint_updated_delta_alpha_beta.txt", torch.cat((updated_joint.delta.reshape(1), updated_joint.theta.reshape(-1))).reshape(-1, 1))

    param0 = {"gamma": gamma, "delta": delta, "theta": theta.reshape(-1, 1)}
    ind_workers = [IndWorker(i + 1, IndLocal(chunk, onedif_kernel, knots))
                   for i, chunk in enumerate(chunks)]
    ind = IndServer(ind_workers, IndGlobal(onedif_kernel, 1., "P"), type_LR="P",
                    if_log=False, logger=None)
    ind_gamma_locals = []
    for worker_id, chunk in enumerate(chunks):
        local = IndLocal(chunk, onedif_kernel, knots); local.type_LR = "P"
        value = local.compute_local_for_gamma(IndParameter(**param0))
        ind_gamma_locals.append(value)
        write_matrix(OUT / f"ind_initial_worker_{worker_id}_gamma.txt", value["gamma"])
        write_matrix(OUT / f"ind_initial_worker_{worker_id}_xx.txt", value["XX"])
    ind_mean_gamma = sum(v["gamma"] for v in ind_gamma_locals) / workers
    ind_mean_xx = sum(v["XX"] for v in ind_gamma_locals) / workers
    ind_gamma_after = IndGlobal(onedif_kernel, 1., "P").update_gamma(
        IndParameter(**param0), {"gamma": ind_mean_gamma, "XX": ind_mean_xx})
    write_matrix(OUT / "ind_initial_global_gamma.txt", ind_mean_gamma)
    write_matrix(OUT / "ind_initial_global_xx.txt", ind_mean_xx)
    write_matrix(OUT / "ind_initial_gamma_new.txt", ind_gamma_after.gamma)
    ind_joint_locals = []
    for worker_id, chunk in enumerate(chunks):
        local = IndLocal(chunk, onedif_kernel, knots); local.type_LR = "P"
        value = local.compute_local_for_theta_delta(ind_gamma_after)
        ind_joint_locals.append(value)
        ind_err = local.local_errorv(ind_gamma_after.gamma)
        ind_R = local.local_cov(ind_gamma_after.theta)
        ind_R.diagonal().add_(1 / ind_gamma_after.delta)
        ind_W = torch.linalg.inv(ind_R)
        ind_obj = (torch.logdet(ind_R) + ind_err.T @ ind_W @ ind_err).squeeze() / n_local
        write_matrix(OUT / f"ind_initial_worker_{worker_id}_objective.txt", ind_obj.reshape(1,1))
        write_matrix(OUT / f"ind_initial_worker_{worker_id}_gradient.txt", value["grad"].reshape(-1,1))
        write_matrix(OUT / f"ind_initial_worker_{worker_id}_hessian.txt", value["hessian"])
    ind.run(IndParameter(**param0), T=rounds, method="direct")
    record_trajectory(OUT / "ind_trajectory.csv", ind.params)

    lr_workers = [LrWorker(i + 1, LrLocal(chunk, onedif_kernel, knots))
                  for i, chunk in enumerate(chunks)]
    lr = LrServer(lr_workers, LrGlobal(knots, onedif_kernel, 1.), type_LR="P",
                  dl_th_together=True, if_log=False, logger=None)
    lr.run(LrParameter(**param0), T=rounds, method="direct")
    record_trajectory(OUT / "low_rank_trajectory.csv", lr.params_GDT)
    write_matrix(OUT / "p_iteration1_mu.txt", lr.params_GDT[1].mu)
    write_matrix(OUT / "p_iteration1_sigma.txt", lr.params_GDT[1].sigma)

    (OUT / "metadata.json").write_text(json.dumps({
        "provenance": "Python reference fixture only; never production input",
        "figure": 1, "seed": seed, "nu": nu, "beta": beta, "alpha": 1.,
        "noise_sd": 2., "delta": .25, "workers": workers, "n_local": n_local,
        "m": m, "rounds": rounds, "coefficients": coefficients,
        "partition": "consecutive torch.chunk", "knots": "jittered grid",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
