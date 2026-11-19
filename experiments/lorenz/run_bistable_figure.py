"""The numbers of the bistable Lorenz figure of the section "Bistable Dynamics with an Unstable Initial Guess".

Every number in the figure is measured here; the figure itself is drawn by the notebook
experiments/lorenz/figure_bistable.ipynb (builder:
experiments/lorenz/build_figure_bistable_notebook.py) from the JSON and the npz written here. The system is built with
`make_system()` of experiments/lorenz/run_benchmark.py, so the figure and the
benchmark solve the same problem: Lorenz at sigma = 10, rho = 10, beta = 8/3, x0 = (1, 1, 1),
explicit Euler at dt = 0.01 with N = 1000 steps in fp64, and the benchmark's zero guess.

Panels (one text width, three panels):
  left    the two basins. The origin is a saddle and C+- = (+-sqrt(beta(rho-1)), same, rho-1)
          are stable foci. The trajectory from x0 settles at C+, its mirror image at C-, and the
          zero guess is the constant trajectory at the saddle. The horizon exponents
          lambda_h = log||J_{N-1} ... J_1||_2 / ((N-1) dt) of the guess and of the solution are
          computed here (parode.laser.lle_dense).
  centre  contours of the merit functional on the plane spanned by the smallest and largest right
          singular vectors of the residual Jacobian at the guess, without damping and with the
          LASER damping J_t -> exp(-gamma dt) J_t, gamma = ReLU(lambda_h + eps), eps = 1/T. The
          damped panel is the merit functional of the anchored recurrence
          x_t = damping g(x_{t-1}) + (1 - damping) g(x^k_{t-1}) with the damping factor
          damping = exp(-gamma dt), whose Newton step is the damped step and whose fixed point is the same
          trajectory. (rho is the Lorenz parameter, 10. The JSON stores the damping factor as rho_damping and
          under the key "rho" of its entries "undamped" and "damped", which keep that name.)
          The field plotted is the quadratic (Gauss-Newton) model of that merit at the guess,
          L(d) = ||r + A d||^2 / (2 dt), which is what the Newton step minimizes and what kappa
          measures. The true merit is computed on the same plane and kept in the cache, but it does
          not show the conditioning: v_min is a near-null direction of A, so along it the quartic
          term of the Lorenz nonlinearity overtakes the quadratic term (sigma_min beta)^2 already at
          a state displacement of about 0.5, and both panels would read as flat valleys.
  right   max-norm residual against Newton iteration for plain DEER, constant scale-ELK at
          k = 0.05, ELK at a constant sigma^2, and LASER (the damping proposed before the first trial). ELK runs
          at sigma^2 = 1e2 and 1e4; the notebook draws the one that converges in fewer iterations
          (elk_best_sigmasq, 1e4). The residual figure of the appendix uses other constants, the best of the
          tuning sweep (run_residual_histories.py). Non-finite iterates and runs that reach the iteration cap
          are marked, not dropped.

Linear algebra of the centre panel. The residual Jacobian A = dr/dX is block lower bidiagonal,
identity diagonal and -damping J_t subdiagonal, of size N D = 3000. We do not take an SVD of A to get
sigma_min: at the guess sigma_min is 8e-25 and a dense SVD of A returns rounding noise (6e-17) for
it. We use the structure instead. sigma_max(A) and its right singular vector come from ARPACK
(scipy.sparse.linalg.svds) on a matrix-free LinearOperator whose matvec is one bidiagonal sweep.
sigma_min(A) = 1 / sigma_max(A^{-1}) and its right singular vector is the leading LEFT singular
vector of A^{-1}, again from ARPACK on a matrix-free operator: A^{-1} applied to b is the forward
recurrence X_t = damping J_{t-1} X_{t-1} + b_t (the sequential form of the DEER scan) and its transpose
is the backward sweep. Both cost O(N D^2). A dense cross-check (explicit A and A^{-1}, numpy SVD)
is run by default and reported in the JSON.

Run (CPU batch job, a few minutes):
  python experiments/lorenz/run_bistable_figure.py
It writes into results/lorenz/check/bistable_figure/; --outdir results/lorenz/bistable_figure replaces the shipped
numbers that the notebook reads.
"""
import argparse
import json
import logging
import os
import sys
import time

logging.getLogger("jax._src.xla_bridge").setLevel(logging.CRITICAL)

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse.linalg as spla

jax.config.update("jax_enable_x64", True)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from experiments.lorenz.run_benchmark import make_system, scale_elk_step   # noqa: E402
from parode.elk_baseline import elk_step                       # noqa: E402
from parode.explicit_parallel import (                                   # noqa: E402
    affine_terms, euler_step, explicit_par_integrate, linear_terms, par_init, par_scan,
)
from parode.laser import LaserConfig, lle_dense                        # noqa: E402
from parode.ode_utils import prepare_ode_function                        # noqa: E402
from parode.sequential import seq_integrate                              # noqa: E402

OUT = os.path.join(ROOT, "results", "lorenz", "check", "bistable_figure")
TOL = 1e-9
MAX_ITER = 200


# ----------------------------------------------------------------------------- the system

def build(N, dt):
    """The benchmark's Lorenz configuration: ode, x0, the sequential truth and the zero guess."""
    ode, x0, label = make_system()
    ode_fn = prepare_ode_function(ode)
    x0 = jnp.asarray(x0, jnp.float64)
    truth = seq_integrate(ode, x0, dt, N, "euler")
    guess = jnp.zeros((N, 3), x0.dtype)
    times = dt * jnp.arange(N + 1)
    return ode, ode_fn, x0, truth, guess, times, label


def exponent(ode_fn, X, dt, times):
    """lambda_h(X) = log ||J_{N-1} ... J_1||_2 / ((N-1) dt) for the Euler step Jacobians along X."""
    F = linear_terms(X, ode_fn, dt, euler_step, times, None)
    return float(lle_dense(F, dt))


# ----------------------------------------------------------------------------- merit functional

def make_merit(ode_fn, x0, guess, dt, times):
    """Return resid(X, damping) and merit(X, damping) for the anchored recurrence at anchor `guess`.

    damping = 1 is the plain residual r(X)_t = X_t - g(X_{t-1}) (g the Euler step, X_0 = x0), and
    damping < 1 is the residual of x_t = damping g(x_{t-1}) + (1 - damping) g(x^k_{t-1}) with x^k = guess.
    The two agree at X = guess, and the Jacobian of the second is I on the diagonal and
    -damping J_t on the subdiagonal. The merit is the Riemann sum of (1/2) int ||s' - f(s)||^2 dt
    under the same Euler quadrature, (1/2) sum_t ||r_t||^2 / dt.
    """
    def g(Y):
        return Y + jax.vmap(euler_step, in_axes=(None, 0, 0, None, None))(ode_fn, Y, times[:-1], dt, None)

    g_anchor = g(jnp.vstack((x0, guess[:-1])))

    def resid(X, damping):
        return X - damping * g(jnp.vstack((x0, X[:-1]))) - (1.0 - damping) * g_anchor

    def merit(X, damping):
        r = resid(X, damping)
        return 0.5 * jnp.sum(r * r) / dt

    return resid, merit


# ----------------------------------------------------------------------------- conditioning

def sparse_ops(Fs):
    """Matrix-free A and A^{-1} for the block lower bidiagonal A (I on the diagonal, -Fs below).

    Fs has shape (N-1, D, D) and is already scaled by the damping factor. A matvec is one sweep, O(N D^2).
    """
    N1, D = Fs.shape[0], Fs.shape[1]
    N = N1 + 1
    n = N * D

    def mv(x):
        X = x.reshape(N, D).copy()
        X[1:] -= np.einsum("tij,tj->ti", Fs, x.reshape(N, D)[:-1])
        return X.ravel()

    def rmv(x):
        X = x.reshape(N, D).copy()
        X[:-1] -= np.einsum("tji,tj->ti", Fs, x.reshape(N, D)[1:])
        return X.ravel()

    def mv_inv(b):
        B = b.reshape(N, D)
        X = np.empty_like(B)
        X[0] = B[0]
        for t in range(1, N):
            X[t] = Fs[t - 1] @ X[t - 1] + B[t]
        return X.ravel()

    def rmv_inv(b):
        B = b.reshape(N, D)
        Y = np.empty_like(B)
        Y[-1] = B[-1]
        for t in range(N - 2, -1, -1):
            Y[t] = B[t] + Fs[t].T @ Y[t + 1]
        return Y.ravel()

    A = spla.LinearOperator((n, n), matvec=mv, rmatvec=rmv, dtype=Fs.dtype)
    Ainv = spla.LinearOperator((n, n), matvec=mv_inv, rmatvec=rmv_inv, dtype=Fs.dtype)
    return A, Ainv


def dense_blocks(Fs):
    """Explicit A and A^{-1} (the prefix products of Fs), for the cross-check only."""
    N1, D = Fs.shape[0], Fs.shape[1]
    N = N1 + 1
    A = np.zeros((N * D, N * D))
    Ainv = np.zeros((N * D, N * D))
    eye = np.eye(D)
    for t in range(N):
        A[t * D:(t + 1) * D, t * D:(t + 1) * D] = eye
        if t > 0:
            A[t * D:(t + 1) * D, (t - 1) * D:t * D] = -Fs[t - 1]
    prev = eye[None]
    Ainv[0:D, 0:D] = eye
    for t in range(1, N):
        row = np.concatenate([np.einsum("ij,sjk->sik", Fs[t - 1], prev), eye[None]], 0)
        Ainv[t * D:(t + 1) * D, :(t + 1) * D] = row.transpose(1, 0, 2).reshape(D, (t + 1) * D)
        prev = row
    return A, Ainv


def conditioning(Fs, dense_check=True, seed=0):
    """(sigma_max, sigma_min, v_max, v_min, info) of the residual Jacobian with subdiagonal -Fs.

    Primary route: ARPACK on the matrix-free operators. sigma_min comes from sigma_max(A^{-1})
    because a direct SVD of A cannot resolve it (see the module docstring).
    """
    rng = np.random.default_rng(seed)
    A, Ainv = sparse_ops(Fs)
    n = A.shape[0]
    x, y = rng.standard_normal(n), rng.standard_normal(n)
    info = {"adjoint_err": float(abs(x @ (A.rmatvec(y)) - (A.matvec(x)) @ y) / (np.linalg.norm(x) * np.linalg.norm(y))),
            "inverse_err": float(np.linalg.norm(A.matvec(Ainv.matvec(y)) - y) / np.linalg.norm(y))}
    t0 = time.perf_counter()
    v0 = rng.standard_normal(n)
    u1, s1, vt1 = spla.svds(A, k=1, which="LM", v0=v0)
    u2, s2, vt2 = spla.svds(Ainv, k=1, which="LM", v0=v0)
    info["arpack_s"] = time.perf_counter() - t0
    # A = sum sigma_i u_i v_i^T. The leading triplet of A gives (sigma_max, v_max, u_max) directly.
    # For A^{-1} = sum s_i p_i q_i^T the leading triplet gives A^{-1} q = s p, hence A p = q / s:
    # v_min = p = u2[:, 0], sigma_min = 1 / s, and its image direction is u_min = q = vt2[0]. Taking
    # A v_min by a matvec instead would return rounding noise when sigma_min is below the fp64 floor.
    smax, v_max, u_max = float(s1[0]), np.asarray(vt1[0]), np.asarray(u1[:, 0])
    smin, v_min, u_min = 1.0 / float(s2[0]), np.asarray(u2[:, 0]), np.asarray(vt2[0])
    info["resid_max"] = float(np.linalg.norm(A.matvec(v_max)) / smax - 1.0)
    info["resid_min_direct_matvec"] = float(np.linalg.norm(A.matvec(v_min)))
    info["vmax_vmin_overlap"] = float(abs(v_max @ v_min))
    info["umax_umin_overlap"] = float(abs(u_max @ u_min))
    if dense_check:
        t0 = time.perf_counter()
        Ad, Ainvd = dense_blocks(Fs)
        sd = np.linalg.svd(Ad, compute_uv=False)
        sid = np.linalg.svd(Ainvd, compute_uv=False)
        info["dense_s"] = time.perf_counter() - t0
        info["dense_sigma_max"] = float(sd[0])
        info["dense_sigma_min"] = float(1.0 / sid[0])
        info["dense_sigma_min_direct"] = float(sd[-1])      # rounding noise when kappa > 1/eps
        info["dense_rel_err_sigma_max"] = float(abs(sd[0] - smax) / sd[0])
        info["dense_rel_err_sigma_min"] = float(abs(1.0 / sid[0] - smin) * sid[0])
    return smax, smin, v_max, v_min, u_max, u_min, info


# ----------------------------------------------------------------------------- solvers, per iteration

def scaled_newton_history(ode_fn, x0, guess, dt, times, k, tol=TOL, max_iter=MAX_ITER):
    """Residual max|r| after each iteration of constant scale-ELK (F -> (1 - k) F); k = 0 is DEER."""
    @jax.jit
    def step(X):
        return scale_elk_step(X, x0, ode_fn, dt, times, k)

    c0 = affine_terms(guess, x0, ode_fn, dt, euler_step, times, None)
    res = [float(jnp.max(jnp.abs(c0)))]
    X = guess
    for _ in range(max_iter):
        X, r = step(X)
        res.append(float(r))
        if not np.isfinite(res[-1]) or res[-1] < tol:
            break
    return np.array(res)


def elk_history(ode_fn, x0, guess, dt, times, sigmasq, tol=TOL, max_iter=MAX_ITER):
    """Residual after each ELK iteration at a fixed sigma^2 (dense filter, Gonzalez et al. 2024)."""
    step = jax.jit(lambda X: elk_step(X, x0, ode_fn, dt, times, sigmasq, False))
    c0 = affine_terms(guess, x0, ode_fn, dt, euler_step, times, None)
    res = [float(jnp.max(jnp.abs(c0)))]
    X = guess
    for _ in range(max_iter):
        X, r = step(X)
        res.append(float(r))
        if not np.isfinite(res[-1]) or res[-1] < tol:
            break
    return np.array(res)


def laser_history(ode, x0, guess, dt, tol=TOL, max_iter=MAX_ITER):
    """LASER: the paper's configuration (the defaults), per-iteration residual, gamma, lambda_h and trials."""
    cfg = LaserConfig(return_history=True)

    def solve(x0, guess):                # under jax.jit, as in run_benchmark: eager parode.laser code hangs on a GPU node
        c0 = jnp.max(jnp.abs(affine_terms(guess, x0, prepare_ode_function(ode), dt, euler_step,
                                          dt * jnp.arange(guess.shape[0] + 1), None)))
        return explicit_par_integrate(ode, x0, guess, dt, max_iter, "euler", tol=tol, laser=cfg), c0

    (sol, res, it, hist), c0 = jax.jit(solve)(x0, guess)
    it = int(it)
    h = {k: np.asarray(v)[:it] for k, v in hist.items()}
    return np.concatenate(([float(c0)], h["residual"])), h, np.asarray(sol), float(res), it


# ----------------------------------------------------------------------------- measurement

def measure(N=1000, dt=0.01, grid=81, win=2.0, dense_check=True, elk_sigmasq=(100.0, 10000.0)):
    """Compute every quantity in the figure. Returns (json-serializable dict, arrays for the plot)."""
    ode, ode_fn, x0, truth, guess, times, label = build(N, dt)
    sigma, rho_l, beta_l = 10.0, 10.0, 8.0 / 3.0
    q = {"system": label, "N": N, "dt": dt, "T": N * dt, "D": 3, "dtype": "fp64",
         "x0": np.asarray(x0).tolist(), "sigma": sigma, "rho": rho_l, "beta": beta_l,
         "tol": TOL, "max_iter": MAX_ITER, "guess": "zeros"}

    # --- fixed points and their linearizations -------------------------------------------------
    f = lambda x: ode_fn(0.0, x, None)                                       # noqa: E731
    c_star = float(np.sqrt(beta_l * (rho_l - 1.0)))
    Cp, Cm = jnp.array([c_star, c_star, rho_l - 1.0]), jnp.array([-c_star, -c_star, rho_l - 1.0])
    q["C_plus"] = np.asarray(Cp).tolist()
    q["C_minus"] = np.asarray(Cm).tolist()
    for name, p in (("origin", jnp.zeros(3)), ("C_plus", Cp)):
        ev = np.linalg.eigvals(np.asarray(jax.jacfwd(f)(p)))
        ev = ev[np.argsort(-ev.real)]
        q[f"eig_{name}"] = [[float(v.real), float(v.imag)] for v in ev]
    q["saddle_unstable_rate"] = float(max(np.linalg.eigvals(np.asarray(jax.jacfwd(f)(jnp.zeros(3)))).real))

    # --- trajectories and horizon exponents ----------------------------------------------------
    x0_mirror = jnp.array([-1.0, -1.0, 1.0])
    truth_mirror = seq_integrate(ode, x0_mirror, dt, N, "euler")
    q["x0_mirror"] = np.asarray(x0_mirror).tolist()
    q["truth_final"] = np.asarray(truth[-1]).tolist()
    q["truth_mirror_final"] = np.asarray(truth_mirror[-1]).tolist()
    q["truth_max_abs"] = float(np.max(np.abs(np.asarray(truth))))
    q["lambda_h_guess"] = exponent(ode_fn, guess, dt, times)
    q["lambda_h_truth"] = exponent(ode_fn, truth[1:], dt, times)
    q["lambda_h_truth_mirror"] = exponent(ode_fn, truth_mirror[1:], dt, times)
    q["eps"] = 1.0 / (N * dt)
    q["gamma"] = float(max(q["lambda_h_guess"] + q["eps"], 0.0))
    q["rho_damping"] = float(np.exp(-q["gamma"] * dt))
    q["lambda_h_guess_damped"] = q["lambda_h_guess"] - q["gamma"]

    # --- conditioning of the merit functional at the guess -------------------------------------
    F0 = np.asarray(linear_terms(guess, ode_fn, dt, euler_step, times, None))
    resid, merit = make_merit(ode_fn, x0, guess, dt, times)
    c_check = np.asarray(affine_terms(guess, x0, ode_fn, dt, euler_step, times, None))
    r_check = np.asarray(resid(guess, 1.0))
    q["residual_definition_max_err"] = float(np.max(np.abs(r_check + c_check)))
    R2 = float(np.linalg.norm(r_check))
    q["residual_2norm_at_guess"] = R2
    q["residual_maxnorm_at_guess"] = float(np.max(np.abs(r_check)))

    panels = {}
    for tag, damping in (("undamped", 1.0), ("damped", q["rho_damping"])):
        smax, smin, vmax, vmin, umax, umin, info = conditioning(damping * F0, dense_check=dense_check)
        panels[tag] = dict(damping=damping, sigma_max=smax, sigma_min=smin, kappa=smax / smin,
                           vmax=vmax, vmin=vmin, umax=umax, umin=umin, info=info)
        # "rho" in these two entries is the damping factor, the name the shipped JSON has (q["rho"] is Lorenz's rho)
        q[tag] = {"rho": damping, "sigma_max": smax, "sigma_min": smin, "kappa": smax / smin,
                  "log10_kappa": float(np.log10(smax / smin)), **{k: v for k, v in info.items()}}

    # --- merit contours on the (v_max, v_min) plane, same window in both panels -----------------
    half_a = win * R2 / panels["damped"]["sigma_max"]
    half_b = win * R2 / panels["damped"]["sigma_min"]
    q["window"] = {"half_alpha": half_a, "half_beta": half_b, "win_factor": win, "grid": grid,
                   "rule": "half_alpha = win ||r||_2 / sigma_max(damped), half_beta = win ||r||_2 / sigma_min(damped)"}
    alphas = np.linspace(-half_a, half_a, grid)
    betas = np.linspace(-half_b, half_b, grid)
    grids, grids_true = {}, {}
    r0 = r_check.ravel()
    for tag in ("undamped", "damped"):
        p = panels[tag]
        Vmax = jnp.asarray(p["vmax"].reshape(N, 3))
        Vmin = jnp.asarray(p["vmin"].reshape(N, 3))
        damping = p["damping"]

        @jax.jit
        def row(a, bs, Vmax=Vmax, Vmin=Vmin, damping=damping):
            def one(b):
                X = guess + a * Vmax + b * Vmin
                return merit(X, damping)
            return jax.vmap(one)(bs)

        # True merit on the plane, kept for the record (see q["centre_panel"]).
        L_true = np.stack([np.asarray(row(float(a), jnp.asarray(betas))) for a in alphas])
        # Plotted field: the quadratic (Gauss-Newton) model of the same merit at the guess,
        # L(d) = ||r + A d||^2 / (2 dt), which is the functional Newton minimizes at this iterate and
        # the one whose condition number is kappa. On the plane A d = alpha sigma_max u_max +
        # beta sigma_min u_min with u_max orthogonal to u_min, so the model is available in closed form.
        sm, sn = p["sigma_max"], p["sigma_min"]
        pm, pn = float(r0 @ p["umax"]), float(r0 @ p["umin"])
        cross = 2.0 * float(p["umax"] @ p["umin"]) * sm * sn
        Al, Be = np.meshgrid(alphas, betas, indexing="ij")
        L_gn = (float(r0 @ r0) + 2.0 * Al * sm * pm + 2.0 * Be * sn * pn
                + (Al * sm) ** 2 + (Be * sn) ** 2 + cross * Al * Be) / (2.0 * dt)
        q[tag]["merit_at_guess"] = float(merit(guess, damping))
        q[tag]["model_min_alpha"] = float(-pm / sm)
        q[tag]["model_min_beta"] = float(-pn / sn)
        q[tag]["model_merit_at_guess"] = float(L_gn[grid // 2, grid // 2])
        grids[tag], grids_true[tag] = L_gn, L_true
    q["centre_panel"] = {
        "field": "gauss_newton_model",
        "definition": "L(d) = ||r(X) + A d||^2 / (2 dt) on d = alpha v_max + beta v_min at X = guess",
        "why_not_the_true_merit": (
            "The true merit on this plane is dominated by the Lorenz nonlinearity along v_min rather "
            "than by the conditioning: v_min is a near-null direction of A, so the quadratic term "
            "(sigma_min beta)^2 is overtaken by the quartic term of the true merit at beta of order 14, "
            "a state displacement of about 0.5. The true grids are kept in the cache as L_true_*."),
        "true_merit_grid_min": {t: float(np.min(grids_true[t])) for t in grids_true},
        "true_merit_grid_max": {t: float(np.max(grids_true[t])) for t in grids_true},
    }
    q["merit_grid_min"] = {t: float(np.min(grids[t])) for t in grids}
    q["merit_grid_max"] = {t: float(np.max(grids[t])) for t in grids}
    q["state_displacement"] = {t: {"alpha_edge_max_abs": float(half_a * np.max(np.abs(panels[t]["vmax"]))),
                                   "beta_edge_max_abs": float(half_b * np.max(np.abs(panels[t]["vmin"])))}
                               for t in panels}

    # --- convergence histories -----------------------------------------------------------------
    hist = {}
    hist["deer"] = scaled_newton_history(ode_fn, x0, guess, dt, times, 0.0)
    hist["scale_elk_0.05"] = scaled_newton_history(ode_fn, x0, guess, dt, times, 0.05)
    for s2 in elk_sigmasq:
        hist[f"elk_{s2:g}"] = elk_history(ode_fn, x0, guess, dt, times, s2)
    res_laser, h_laser, sol_laser, _, _ = laser_history(ode, x0, guess, dt)
    hist["laser"] = res_laser

    def status(r, cap=MAX_ITER):
        if not np.isfinite(r[-1]):
            return "nonfinite"
        if r[-1] < TOL:
            return "converged"
        return "max_iter" if len(r) - 1 >= cap else "stalled"

    q["runs"] = {}
    for name, r in hist.items():
        q["runs"][name] = {"iterations": int(len(r) - 1), "final_residual": float(r[-1]),
                           "status": status(r), "residual": [float(v) for v in r]}
    best = min((s for s in elk_sigmasq), key=lambda s: (q["runs"][f"elk_{s:g}"]["status"] != "converged",
                                                        q["runs"][f"elk_{s:g}"]["iterations"]))
    q["elk_best_sigmasq"] = float(best)
    q["elk_sigmasq_tried"] = [float(s) for s in elk_sigmasq]
    q["runs"]["laser"]["trials"] = int(np.sum(h_laser["n_trials"]))
    q["runs"]["laser"]["rejected"] = int(np.sum(h_laser["n_trials"] - 1))
    q["runs"]["laser"]["gamma"] = [float(v) for v in h_laser["gamma"]]
    q["runs"]["laser"]["lambda_h"] = [float(v) for v in h_laser["lambda_h"]]
    q["runs"]["laser"]["err_vs_sequential"] = float(np.max(np.abs(np.asarray(sol_laser) - np.asarray(truth))))
    q["runs"]["deer"]["first_step_residual"] = float(hist["deer"][1])
    c_g = affine_terms(guess, x0, ode_fn, dt, euler_step, times, None)
    F_g = linear_terms(guess, ode_fn, dt, euler_step, times, None)
    _, cs_g = par_scan(par_init(F_g, c_g[1:], c_g[0]))
    X1 = guess + jnp.vstack((c_g[0], cs_g))
    q["runs"]["deer"]["first_step_max_abs_iterate"] = float(jnp.max(jnp.abs(X1)))

    arrays = dict(truth=np.asarray(truth), truth_mirror=np.asarray(truth_mirror),
                  alphas=alphas, betas=betas,
                  L_undamped=grids["undamped"], L_damped=grids["damped"],
                  L_true_undamped=grids_true["undamped"], L_true_damped=grids_true["damped"],
                  **{f"res_{k}": v for k, v in hist.items()},
                  laser_gamma=h_laser["gamma"], laser_lambda=h_laser["lambda_h"],
                  laser_trials=h_laser["n_trials"])
    return q, arrays


# ----------------------------------------------------------------------------- entry point

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--grid", type=int, default=81)
    ap.add_argument("--win", type=float, default=2.0)
    ap.add_argument("--no-dense-check", action="store_true")
    ap.add_argument("--outdir", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    js = os.path.join(args.outdir, "lorenz_bistable.json")
    cache = os.path.join(args.outdir, "lorenz_bistable_cache.npz")

    t0 = time.perf_counter()
    q, arrays = measure(N=args.N, dt=args.dt, grid=args.grid, win=args.win,
                        dense_check=not args.no_dense_check)
    q["measure_seconds"] = time.perf_counter() - t0
    q["device"] = jax.devices()[0].device_kind
    with open(js, "w") as fh:
        json.dump(q, fh, indent=1, sort_keys=False)
    np.savez_compressed(cache, **arrays)
    print(f"wrote {js}\nwrote {cache}\nmeasure {q['measure_seconds']:.1f} s")
    for k in ("lambda_h_guess", "lambda_h_truth", "gamma", "rho_damping", "residual_2norm_at_guess"):
        print(f"  {k:26s} {q[k]:.6g}")
    for tag in ("undamped", "damped"):
        d = q[tag]
        print(f"  {tag:9s} sigma_max {d['sigma_max']:.6g}  sigma_min {d['sigma_min']:.6g}  "
              f"kappa {d['kappa']:.6g}  dense sigma_min {d.get('dense_sigma_min', float('nan')):.6g} "
              f"(direct SVD of A: {d.get('dense_sigma_min_direct', float('nan')):.3g})")
    for name, r in q["runs"].items():
        print(f"  {name:16s} iters {r['iterations']:4d}  final {r['final_residual']:.3g}  {r['status']}")


if __name__ == "__main__":
    main()
