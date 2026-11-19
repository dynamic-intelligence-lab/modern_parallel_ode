"""Spectra of the Lorenz rho sweep, the source of the exponents quoted in the paper (its rho = 10 rows).

For rho in {5, 10, 15, 20, 23}, sigma 10, beta 8/3, explicit Euler dt 0.01, fp64, writes one row per (rho, N)
to --out, by default results/lorenz/check/spectra.csv (the shipped file is results/lorenz/spectra.csv). The per-rho
columns repeat on each of its rows.

Per rho, from the vector field of experiments/lorenz/run_benchmark.make_system(rho=rho) with
jax.jacfwd (numerical, not closed form):
  eig_M0         eigenvalues of M0 = I + dt A(0), A(0) the Jacobian of the field at the origin
  specrad_M0     spectral radius of M0;  norm2_M0  its spectral norm ||M0||_2
  k_eig          1 - 1/specrad_M0 (the threshold measured from the eigenvalue)
  k_norm         1 - 1/norm2_M0 (the threshold at which (1 - k) M0 is non-expansive in the 2-norm)
  k_derived      1 - 1/(1 + dt lambda_u), lambda_u = (-(sigma + 1) + sqrt((sigma + 1)^2 + 4 sigma (rho - 1)))/2
  eig_A_Cplus, eig_A_Cminus   eigenvalues of the continuous Jacobian A(C+-) at the foci C+- (run_benchmark.lorenz_focus)
  eig_M_Cplus    eigenvalues of I + dt A(C+); specrad_M_Cplus and norm2_M_Cplus its spectral radius and norm.
                 specrad_M_Cplus > 1 means the focus is an unstable fixed point of the Euler map at this dt,
                 although the continuous focus is stable for rho < 24.74.
  field_Cplus    max|f(C+)|, a check that C+ is an equilibrium
Per (rho, N), the sequential Euler truth from x0 (1, 1, 1) (run_benchmark.sequential_truth, the driver's truth):
  lam_truth, lamT_truth   lambda_h of the Jacobians at the truth states 1 .. N-1 and lambda_h T, T = (N - 1) dt,
                 by the matrix-free power iteration below (exponent_matrix_free: at most 30 iterations,
                 1e-6 relative); pi_iters_truth its iteration count and pi_gap_truth its last log_b - log_a
  dense_lam_truth, svd_lam_truth   the same exponent densely (make_dense_exponent)
  lam_zero, lamT_zero    the same exponent along the zero guess
  verdict        lambda_h T < -1 contractive, in [-1, 1] neutral, > 1 expansive
  truth_end, truth_max, dist_end_focus   final state, max_t |x_t|_inf, and the inf-norm distance of the final state
                 to the nearer focus

Run on a compute node only (imports JAX):  python experiments/lorenz/run_spectra.py
"""
import argparse
import csv
import math
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from experiments.lorenz.run_benchmark import (                                          # noqa: E402
    LORENZ_BETA, LORENZ_SIGMA, lorenz_focus, make_system, sequential_truth,
)
from parode.explicit_parallel import euler_step, linear_terms                    # noqa: E402
from parode.laser import lle_dense, scaled_product                             # noqa: E402

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
DT = 0.01
RHOS = (5.0, 10.0, 15.0, 20.0, 23.0)
NS = (1000, 10000, 100000)
OUT = os.path.join(ROOT, "results", "lorenz", "check", "spectra.csv")
POWER_MAX_ITER = 30
POWER_RTOL = 1e-6


def make_power_iteration(ode_fn, dt):
    """Jitted one power iteration on A^T A, A = J_{n} ... J_1 along a trajectory X (n, D) at times ts (n,).

    Returns (w, log_a, log_b): log_a = log ||A v|| for the unit input v, log_b = log ||A^T u|| for the
    unit u = A v / ||A v||, and w the next unit start vector. log_a <= log_b <= log sigma_max.
    """
    def step(x, t):
        return x + dt * ode_fn(t, x, None)

    def normalise(y, logn):
        n = jnp.linalg.norm(y)
        return y / jnp.where(n > 0, n, 1.0), logn + jnp.log(n)

    def fwd(carry, xt):
        v, logn = carry
        x, t = xt
        _, jv = jax.jvp(lambda y: step(y, t), (x,), (v,))
        return normalise(jv, logn), None

    def bwd(carry, xt):
        w, logn = carry
        x, t = xt
        _, vjp_fn = jax.vjp(lambda y: step(y, t), x)
        return normalise(vjp_fn(w)[0], logn), None

    @jax.jit
    def one_iteration(v, X, ts):
        zero = jnp.zeros((), X.dtype)
        (u, log_a), _ = lax.scan(fwd, (v, zero), (X, ts))
        (w, log_b), _ = lax.scan(bwd, (u, zero), (X, ts), reverse=True)
        return w, log_a, log_b

    return one_iteration


def exponent_matrix_free(one_iteration, X, ts, dt):
    """lambda_h of the Jacobians along X by power iteration. Returns (lambda_h, iterations, log_b - log_a)."""
    D = X.shape[1]
    v = jax.random.normal(jax.random.PRNGKey(123), (D,))
    v = v / jnp.linalg.norm(v)
    prev, it, est, gap = None, 0, float("nan"), float("nan")
    for it in range(1, POWER_MAX_ITER + 1):
        v, log_a, log_b = one_iteration(v, X, ts)
        est, gap = float(log_b), float(log_b - log_a)
        if not math.isfinite(est):
            break
        if prev is not None and abs(est - prev) < POWER_RTOL * abs(est):
            break
        prev = est
    return est / (X.shape[0] * dt), it, gap


def make_dense_exponent(ode_fn, dt):
    """Jitted dense exponents of the Jacobians at X[:-1]: (lle_dense, SVD of the product)."""
    @jax.jit
    def dense(X, times):
        F = linear_terms(X, ode_fn, dt, euler_step, times, None)
        M_hat, log_s = scaled_product(F)
        svd = (log_s + jnp.log(jnp.linalg.norm(M_hat, 2))) / (F.shape[0] * dt)
        return lle_dense(F, dt), svd
    return dense


def verdict(lam, T):
    if not math.isfinite(lam):
        return "truth diverged"
    x = lam * T
    return "contractive" if x < -1.0 else ("expansive" if x > 1.0 else "neutral")


def cstr(z):
    """Eigenvalues as space-separated a+bi strings, sorted by real part then imaginary part."""
    z = sorted(np.asarray(z).astype(complex), key=lambda w: (w.real, w.imag))
    return " ".join(f"{w.real:.10g}{w.imag:+.10g}i" for w in z)


def derived_threshold(rho, sigma=LORENZ_SIGMA):
    lam_u = (-(sigma + 1.0) + math.sqrt((sigma + 1.0) ** 2 + 4.0 * sigma * (rho - 1.0))) / 2.0
    return 1.0 - 1.0 / (1.0 + DT * lam_u), lam_u


def point_spectra(ode_fn, x):
    """(A(x), max|f(x)|) with A the Jacobian of the field at x, jitted as one program."""
    @jax.jit
    def jac(y):
        return jax.jacfwd(lambda z: ode_fn(0.0, z, None))(y), jnp.max(jnp.abs(ode_fn(0.0, y, None)))
    A, f = jac(x)
    return np.asarray(A), float(f)


def rho_columns(ode_fn, rho):
    I = np.eye(3)
    A0, _ = point_spectra(ode_fn, jnp.zeros(3))
    M0 = I + DT * A0
    ev0 = np.linalg.eigvals(M0)
    specrad0, norm0 = float(np.max(np.abs(ev0))), float(np.linalg.norm(M0, 2))
    k_der, lam_u = derived_threshold(rho)
    cols = dict(rho=rho, sigma=LORENZ_SIGMA, beta=LORENZ_BETA, dt=DT, eig_M0=cstr(ev0), specrad_M0=specrad0,
                norm2_M0=norm0, M0_symmetric=bool(np.allclose(M0, M0.T)), k_eig=1.0 - 1.0 / specrad0,
                k_norm=1.0 - 1.0 / norm0, lambda_u_derived=lam_u, k_derived=k_der)
    for name, s in (("Cplus", 1), ("Cminus", -1)):
        C = lorenz_focus(rho, s)
        A, f = point_spectra(ode_fn, C)
        M = I + DT * A
        evM = np.linalg.eigvals(M)
        cols.update({f"field_{name}": f, f"eig_A_{name}": cstr(np.linalg.eigvals(A)), f"eig_M_{name}": cstr(evM),
                     f"specrad_M_{name}": float(np.max(np.abs(evM))), f"norm2_M_{name}": float(np.linalg.norm(M, 2))})
    return cols


def truth_columns(ode, x0, rho, N, one_iteration, dense):
    truth = sequential_truth(ode, x0, DT, N)
    times = DT * jnp.arange(N + 1)
    X = truth[1:]
    lam, iters, gap = exponent_matrix_free(one_iteration, X[:-1], times[1:-1], DT)
    d_lle, d_svd = dense(X, times)
    Z = jnp.zeros((N, 3))
    lam0, iters0, _ = exponent_matrix_free(one_iteration, Z[:-1], times[1:-1], DT)
    T = (N - 1) * DT
    end = np.asarray(truth[-1])
    dist = min(float(np.max(np.abs(end - np.asarray(lorenz_focus(rho, s))))) for s in (1, -1))
    return dict(N=N, T=T, lam_truth=lam, lamT_truth=lam * T, pi_iters_truth=iters, pi_gap_truth=gap,
                dense_lam_truth=float(d_lle), svd_lam_truth=float(d_svd), lam_zero=lam0, lamT_zero=lam0 * T,
                pi_iters_zero=iters0, verdict=verdict(lam, T),
                truth_end=" ".join(repr(float(a)) for a in end), truth_max=float(np.max(np.abs(np.asarray(truth)))),
                dist_end_focus=dist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    print(f"device {jax.devices()}", flush=True)
    rows = []
    for rho in RHOS:
        ode, x0, label = make_system(rho=rho)
        x0 = x0.astype(jnp.float64)
        ode_fn = ode                                      # already (t, x, u)
        base = rho_columns(ode_fn, rho)
        print(f"{label}: eig M0 {base['eig_M0']}  ||M0|| {base['norm2_M0']:.6f}  k_eig {base['k_eig']:.6f}  "
              f"k_norm {base['k_norm']:.6f}  k_derived {base['k_derived']:.6f}  specrad M(C+) {base['specrad_M_Cplus']:.6f}",
              flush=True)
        one_iteration = make_power_iteration(ode_fn, DT)
        dense = make_dense_exponent(ode_fn, DT)
        for N in NS:
            row = dict(base, **truth_columns(ode, x0, rho, N, one_iteration, dense))
            print(f"  N {N:6d}  lamT {row['lamT_truth']:+.6g} ({row['verdict']}, dense lam {row['dense_lam_truth']:+.6g}, "
                  f"power {row['lam_truth']:+.6g}, it {row['pi_iters_truth']})  lamT zero {row['lamT_zero']:+.6g}  "
                  f"end {row['truth_end']}  dist to focus {row['dist_end_focus']:.3g}", flush=True)
            rows.append(row)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
