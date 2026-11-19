"""LASER: DEER with the step Jacobians damped by the finite-time Lyapunov exponent of the iterate.

Scales the step Jacobians F_t by exp(-gamma dt) inside the DEER scan while
leaving the residual untouched (so fixed points are unchanged), with gamma
chosen every Newton iteration from the finite-time largest Lyapunov exponent
(LLE) of the current iterate, gamma = ReLU(lambda_h + eps). A
non-monotone acceptance test guards the step (growth cap); on rejection lambda
is recovered from the rejected scan and gamma raised; after n_retry rejections
a Jacobi step is taken.

The solver-agnostic loop `run_damped_newton` is called by explicit_par_integrate;
callers supply terms_fn (X -> (F, c) with c = -r) and scan_fn ((F_scaled, c) -> delta),
with dense F (N-1, D, D). Callers whose residual is a separate computation
(explicit) also pass residual_fn: the loop then carries (X, c) only and forms the
Jacobians at the top of each iteration, fused into the scan as in plain DEER; otherwise the
accepted trial's (F, c) are carried, so accepted iterations evaluate the Jacobians once.
The exponent lambda_h is computed exactly by the overflow-free scaled tree product
(`initial_lle`) only where it is needed: on a rejection, and on every accepted
iterate in "lle" mode or when a history is requested. Reading it from the DEER
scan itself (prefix products or probe columns) costs 0.75 to 1 scan on the
GPU, because XLA prunes the unused F products of a plain scan.
"""
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import jax.numpy as jnp
from jax import lax


@dataclass(frozen=True)
class LaserConfig:
    """Static configuration of LASER (hashable, jit-static). The defaults are the configuration that every experiment
    of the paper runs."""
    growth_cap: float = 100.0         # accept iff |r_new| <= growth_cap * reference; 100 lets the transient
                                      #   residual growth of bounded systems through while still rejecting
                                      #   e^{lambda T} blow-ups (growth 1e8 and beyond)
    step_cap: Optional[float] = 2.0   # second growth test per accepted step, |r_new| <= step_cap |r_old|,
                                      #   applied together with the growth_cap test (None: off)
    min_block: int = 8                # window length in steps (a power of two) of the window exponent that
                                      #   history_extra records (lambda_w)
    trust_radius: Optional[float] = None  # residual-dependent switch-off: damp the horizon only while the
                                      #   amplified residual e^{lambda L dt} max|r| would exceed trust_radius * max(1, max|X|),
                                      #   i.e. gamma = ReLU(lambda + log(max|r| / Delta) / (L dt) + eps) (None: off)
    accept_ref: str = "best"          # reference of the growth test: "best" = the smallest accepted residual so far
                                      #   (bounds the total excursion by growth_cap; needed on Lorenz, cart-pole and
                                      #   controller systems), "prev" = the current iterate's residual (growth can
                                      #   compound across iterations: a run may drift to infinity in steps of growth_cap)
    nu_up: float = 4.0                # gamma growth factor on rejection
    nu_down: float = 4.0              # hysteresis floor gamma_acc / nu_down (see floor_switch_off)
    floor_switch_off: bool = False    # False: the floor persists and decays by nu_down per accepted iteration,
                                      #   LM-style memory of the last working damping (needed where the truth has a
                                      #   long expansive stretch that the horizon average does not see); True: the
                                      #   floor is dropped after two consecutive contractive accepted iterates (needs
                                      #   the exponent: propose_first "lle" or the history)
    n_retry: int = 8                  # rejections before the Jacobi fallback
    propose_first: str = "lle"        # "lle": gamma = ReLU(lambda_h + eps) every iteration (lambda_h(X0) by one extra scan)
                                      # "plain": undamped Newton first, LLE damping only after a rejection
    exponent: str = "exact"           # "exact": the exact horizon exponent of every accepted iterate by one tree
                                      #   product (the rule as published); "carry": the exponent of the guess (lam0) is
                                      #   carried from iteration to iteration and replaced only by the exact exponent
                                      #   that a rejection's retry computes (_retry_gamma), so an accepted first trial
                                      #   costs the scan, the terms and the acceptance test, like plain DEER plus the
                                      #   scaling. "carry" needs propose_first="lle".
    return_history: bool = False
    history_extra: bool = False       # with return_history: also record per iteration the first trial (its gamma, the
                                      #   largest over steps; whether it was accepted; its residual ratio max|r_new| /
                                      #   max|r_old|), the mean accepted gamma over steps and the largest min_block-window
                                      #   exponent of the new iterate (EXTRA_HISTORY_FIELDS). Recording only: the
                                      #   iterates, residuals and damping are those of history_extra=False

    def __post_init__(self):
        if self.min_block & (self.min_block - 1):
            raise ValueError("need min_block to be a power of two")
        if self.trust_radius is not None and self.trust_radius <= 0:
            raise ValueError("need trust_radius > 0 (or None)")
        if self.min_block < 1:
            raise ValueError("need min_block >= 1")
        if self.propose_first not in ("plain", "lle"):
            raise ValueError(f"propose_first must be 'plain' or 'lle', got {self.propose_first!r}")
        if self.exponent not in ("exact", "carry"):
            raise ValueError(f"exponent must be 'exact' or 'carry', got {self.exponent!r}")
        if self.exponent == "carry" and self.propose_first != "lle":
            raise ValueError("exponent='carry' needs propose_first='lle'")
        if self.accept_ref not in ("prev", "best"):
            raise ValueError(f"accept_ref must be 'prev' or 'best', got {self.accept_ref!r}")
        if self.n_retry < 0 or self.nu_up < 1.0 or self.nu_down < 1.0 or self.growth_cap < 1.0:
            raise ValueError("need n_retry >= 0, nu_up >= 1, nu_down >= 1, growth_cap >= 1")
        if self.step_cap is not None and self.step_cap < 1.0:
            raise ValueError("need step_cap >= 1 (or None)")


# ---------------------------------------------------------------------------
# Norms and finite-time Lyapunov exponents
# ---------------------------------------------------------------------------

POWER_ITERATIONS = 16


def spectral_norm(P: jnp.ndarray, n_iter: int = POWER_ITERATIONS) -> jnp.ndarray:
    """Largest singular value of a (..., D, D) matrix by power iteration on P^T P.

    Matvecs only: a GPU SVD has millisecond latency even for tiny matrices, which
    dominated the solver's per-iteration cost. Converges like (sigma_2/sigma_1)^(2 n_iter);
    for products of many Jacobians the gap is large, and for P proportional to an
    isometry every start vector is already exact.
    """
    D = P.shape[-1]
    if D == 1:
        return jnp.abs(P[..., 0, 0])
    v0 = jnp.ones(P.shape[:-2] + (D,), P.dtype) / jnp.sqrt(D)
    v0 = v0.at[..., 0].multiply(1.0 + 1e-3)                 # break exact symmetry ties

    def body(_, v):
        w = jnp.einsum("...ij,...j->...i", P, v)
        w = jnp.einsum("...ji,...j->...i", P, w)             # P^T (P v)
        n = jnp.sqrt(jnp.sum(w * w, axis=-1, keepdims=True))
        return jnp.where(n > 0, w / n, v)

    v = lax.fori_loop(0, n_iter, body, v0, unroll=True)     # straight-line code: few kernels on GPU
    Pv = jnp.einsum("...ij,...j->...i", P, v)
    return jnp.sqrt(jnp.sum(Pv * Pv, axis=-1))


def log_spectral_norm_squaring(P: jnp.ndarray, squarings: int = 5, n_iter: int = 4) -> jnp.ndarray:
    """log of the largest singular value of a batch (..., D, D), accurate when singular values are close.

    A 16-step power iteration underestimates sigma_1 by a few percent when sigma_2 / sigma_1 is near 1,
    which is the case for products of a few near-identity Jacobians (short windows), and at long horizons
    that error exceeds eps = 1/T and flips the sign of the scaled exponent. Repeated squaring of the Gram
    matrix G = P^T P raises the gap to (sigma_2 / sigma_1)^(2^(squarings + 1)) before a short power
    iteration, at the cost of `squarings` extra D x D products per matrix; the scale is tracked in log
    space so nothing overflows. Non-finite input gives +inf.
    """
    if P.shape[-1] == 1:
        return jnp.log(jnp.abs(P[..., 0, 0]))
    return _log_spectral_norm_squaring_exact(P, squarings, n_iter)


def _log_spectral_norm_squaring_exact(P: jnp.ndarray, squarings: int, n_iter: int) -> jnp.ndarray:
    D = P.shape[-1]
    finite = jnp.all(jnp.isfinite(P), axis=(-2, -1))
    Pf = jnp.where(finite[..., None, None], P, 0.0)
    G = jnp.einsum("...ji,...jk->...ik", Pf, Pf)                       # G = P^T P, eigenvalues sigma_i^2
    log_total = jnp.zeros(P.shape[:-2], P.dtype)                        # log of the scale of G^(2^s)
    for k in range(squarings + 1):
        m = jnp.max(jnp.abs(G), axis=(-2, -1))
        m_safe = jnp.where(m > 0, m, 1.0)
        G = G / m_safe[..., None, None]
        log_total = log_total + jnp.log(m_safe) * (2 ** (squarings - k))
        if k < squarings:
            G = jnp.einsum("...ij,...jk->...ik", G, G)
    # largest eigenvalue of the normalized G^(2^s) (PSD, gap now negligible): a short power iteration
    v = jnp.ones(P.shape[:-2] + (D,), P.dtype) / jnp.sqrt(D)
    for _ in range(n_iter):
        w = jnp.einsum("...ij,...j->...i", G, v)
        n = jnp.sqrt(jnp.sum(w * w, axis=-1, keepdims=True))
        v = jnp.where(n > 0, w / n, v)
    lam = jnp.sum(v * jnp.einsum("...ij,...j->...i", G, v), axis=-1)     # Rayleigh quotient
    log_sigma1 = (jnp.log(jnp.maximum(lam, jnp.finfo(P.dtype).tiny)) + log_total) / (2.0 * 2 ** squarings)
    zero = jnp.max(jnp.abs(Pf), axis=(-2, -1)) == 0
    return jnp.where(finite, jnp.where(zero, -jnp.inf, log_sigma1), jnp.inf)


def safe_lognorm(P: jnp.ndarray) -> jnp.ndarray:
    """log of an overflow-safe norm: spectral norm for a (D, D) matrix, max-abs for a (D,) vector.

    The spectral norm avoids the 0.5 log(D) upward bias a Frobenius norm would put on
    the exponent. Returns +inf if P has infinite entries, -inf if P == 0, NaN if P has NaNs.
    """
    a = jnp.abs(P)
    m = jnp.max(a)
    if P.ndim == 1:
        return jnp.log(m)
    m_safe = jnp.where((m > 0) & jnp.isfinite(m), m, 1.0)
    log_sigma = jnp.log(spectral_norm(jnp.where(jnp.isfinite(P), P, 0.0) / m_safe))
    return jnp.where(jnp.isfinite(m) & (m > 0), jnp.log(m_safe) + log_sigma, jnp.log(m))


def scaled_product(F: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Total product F_N ... F_1 of an (N, D, D) sequence as (M_hat, log_s) with max|M_hat| = 1.

    A tree reduction (N - 1 matmuls, log N levels, one max-abs normalization per node):
    about half the work of the prefix scan and no down-sweep, which is all the exponent
    needs. Overflow-free; a zero product gives log_s = -inf.
    """
    D = F.shape[-1]
    eye = jnp.eye(D, dtype=F.dtype)[None]

    def normalize(P, logs):
        m = jnp.max(jnp.abs(P), axis=(-2, -1))
        m_safe = jnp.where(m > 0, m, 1.0)
        return P / m_safe[:, None, None], logs + jnp.log(m)

    P, logs = normalize(F, jnp.zeros(F.shape[0], F.dtype))
    while P.shape[0] > 1:
        if P.shape[0] % 2:
            P = jnp.concatenate([P, eye])
            logs = jnp.concatenate([logs, jnp.zeros(1, F.dtype)])
        P, logs = normalize(jnp.einsum("nij,njk->nik", P[1::2], P[::2]), logs[1::2] + logs[::2])
    return P[0], logs[0]


def lle_dense(F: jnp.ndarray, dt: float) -> jnp.ndarray:
    """Finite-time LLE of a dense Jacobian sequence (N, D, D): log sigma_max(F_N...F_1) / (N dt), from the
    overflow-free tree product (scaled_product) and the spectral norm of the normalized total product."""
    M_hat, log_s = scaled_product(F)
    return (log_s + safe_lognorm(M_hat)) / (F.shape[0] * dt)


def initial_lle(F: jnp.ndarray, dt: float) -> jnp.ndarray:
    """Horizon LLE of the unscaled step Jacobians (N-1, D, D) of an iterate, as the loop names it: lle_dense."""
    return lle_dense(F, dt)


# ---------------------------------------------------------------------------
# Scaling, schedule and acceptance
# ---------------------------------------------------------------------------

def scale_jacobians(F: jnp.ndarray, gamma: jnp.ndarray, dt: float) -> jnp.ndarray:
    """F_t -> exp(-gamma dt) F_t with a scalar gamma.

    gamma = inf gives the Jacobi step (F' = 0 exactly, even for inf entries of F).
    """
    factor = jnp.exp(-gamma * dt)
    return jnp.where(jnp.isfinite(gamma), factor * F, jnp.zeros_like(F))


# ---------------------------------------------------------------------------
# Window exponents (the lambda_w column of history_extra)
# ---------------------------------------------------------------------------

def window_products(F: jnp.ndarray, window: int):
    """Products of the Jacobians over aligned windows of `window` consecutive steps, overflow-safe.

    Returns (count, lognorm): count[w] is the number of real steps in window w (the last window may be
    partial) and lognorm[w] = log||J_{t+count-1} ... J_t|| over the window (spectral norm) of a dense
    (N-1, D, D) sequence. Computed by pairwise products up to the window length, which is the up-sweep of
    the parallel scan truncated at level log2(window).
    """
    n = F.shape[0]
    n_pad = -(-n // window) * window
    count = jnp.concatenate([jnp.ones(n, F.dtype), jnp.zeros(n_pad - n, F.dtype)]).reshape(-1, window).sum(axis=1)
    D = F.shape[-1]
    P = jnp.concatenate([F, jnp.broadcast_to(jnp.eye(D, dtype=F.dtype), (n_pad - n, D, D))]) if n_pad > n else F
    logs = jnp.zeros(P.shape[0], F.dtype)
    L = 1
    while L < window:                                                  # pairwise products, later factor on the left
        P = jnp.einsum("nij,njk->nik", P[1::2], P[::2]); logs = logs[1::2] + logs[::2]
        m = jnp.max(jnp.abs(P), axis=(-2, -1)); m_safe = jnp.where(m > 0, m, 1.0)
        P = P / m_safe[:, None, None]; logs = logs + jnp.log(m)
        L *= 2
    if L != window:
        raise ValueError("window must be a power of two")
    return count, logs + log_spectral_norm_squaring(P)             # accurate for close singular values


def window_exponents(F: jnp.ndarray, dt: float, window: int = 8, min_block: Optional[int] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Finite-time exponents of the aligned windows: lam_max (largest window exponent) and, per step, the
    exponent of the window containing that step (shape (N-1,)). Any stretch of m steps meets at most
    m/window + 2 windows, so by submultiplicativity lam_max bounds the growth rate of every sub-product
    up to a constant from the two partial windows at its ends. min_block is an alias of window."""
    window = min_block if min_block is not None else window
    n = F.shape[0]
    count, lognorm = window_products(F, window)
    lam = jnp.where(count > 0, lognorm / (jnp.maximum(count, 1.0) * dt), -jnp.inf)
    return jnp.max(lam), jnp.repeat(lam, window)[:n]


# ---------------------------------------------------------------------------
# Schedule and acceptance
# ---------------------------------------------------------------------------

def _floor(gamma_acc, contractive_count, cfg: LaserConfig):
    """Hysteresis floor gamma_acc / nu_down. Dropped after two contractive iterates unless floor_switch_off
    is False."""
    base = gamma_acc / cfg.nu_down
    if not cfg.floor_switch_off:
        return base
    return jnp.where(contractive_count >= 2, 0.0, base)


def _first_gamma(prop, floor, cfg: LaserConfig):
    """First-trial gamma: 0 (plain first) or max(proposal, floor)."""
    proposal = jnp.maximum(prop, floor)
    return jnp.zeros_like(proposal) if cfg.propose_first == "plain" else proposal


def _next_gamma(gamma, prop_rej, eps, floor, cfg: LaserConfig):
    """gamma after a rejected trial: the largest of the (possibly updated) proposal, nu_up gamma, eps
    and the floor."""
    return jnp.maximum(jnp.maximum(prop_rej, cfg.nu_up * gamma), jnp.maximum(eps, floor))


def accept_step(c_old, c_new, X_new, cfg: LaserConfig, res_ref=None) -> jnp.ndarray:
    """Non-monotone acceptance test; a strictly monotone max-norm test must not be used.

    res_ref: reference residual level of the growth test (accept_ref="best": the smallest accepted
    max|r| so far); None uses max|c_old|, the current iterate's residual (accept_ref="prev").
    """
    finite = jnp.all(jnp.isfinite(X_new)) & jnp.all(jnp.isfinite(c_new))
    res_old = jnp.max(jnp.abs(c_old))
    ref = res_old if res_ref is None else res_ref
    res_new = jnp.max(jnp.abs(c_new))
    ok = res_new <= cfg.growth_cap * ref
    if cfg.step_cap is not None:
        ok = ok & (res_new <= cfg.step_cap * res_old)
    return finite & ok


# ---------------------------------------------------------------------------
# The damped Newton loop
# ---------------------------------------------------------------------------

HISTORY_FIELDS = ("lambda_h", "gamma", "residual", "step_inf", "n_trials")
EXTRA_HISTORY_FIELDS = ("gamma_first", "first_ok", "first_ratio", "gamma_mean", "lambda_w")   # cfg.history_extra


def _trial(X, F, c, gamma, trial, prop, dt, cfg, terms_fn, scan_fn, res_ref=None):
    """One damping trial. Returns (ok, X_new, F_new, c_new, step_inf, gamma_eff, prop).

    prop is the a-priori damping proposal of this iteration, passed through; its refinement (the exact
    exponent of the current iterate) is made by `_retry_gamma`, on the retry path only.
    res_ref is the reference level of the growth test (None: the current residual).
    """
    gamma_eff = jnp.where(trial >= cfg.n_retry, jnp.inf, gamma)       # Jacobi fallback
    F_scaled = lax.cond(jnp.any(gamma_eff > 0), lambda: scale_jacobians(F, gamma_eff, dt), lambda: F)
    delta = scan_fn(F_scaled, c)
    X_new = X + delta
    F_new, c_new = terms_fn(X_new)
    ok = accept_step(c, c_new, X_new, cfg, res_ref)
    finite_new = jnp.all(jnp.isfinite(X_new)) & jnp.all(jnp.isfinite(c_new))
    ok = ok | (jnp.all(~jnp.isfinite(gamma_eff)) & finite_new)      # a finite Jacobi step is always accepted
    return ok, X_new, F_new, c_new, jnp.max(jnp.abs(delta)), gamma_eff, prop


def _retry_gamma(gamma_eff, prop_rej, lam_cur, F, floor, eps, dt, n_steps, cfg, flt_max, shift_h=0.0,
                 lam_known=False):
    """Damping of the next trial after a rejected trial with gamma_eff. The exact exponent of
    the current iterate is computed by the overflow-free tree product on the first rejection of the iteration
    (lam_cur NaN) and reused by its later trials; a non-finite exponent (non-finite Jacobians) counts as the
    largest representable rate. lam_known (static): lam_cur is the exact exponent of the current iterate already
    (never NaN), so the guarded recomputation is dropped; under vmap the guard is a select that runs the tree
    product on every retry body otherwise. Returns (lam_cur, prop_rej, gamma)."""
    if not lam_known:
        lam_cur = lax.cond(jnp.isnan(lam_cur), lambda: initial_lle(F, dt), lambda: lam_cur)
    overflow_bound = jnp.log(flt_max) / (n_steps * dt) + gamma_eff
    lam_rej = jnp.where(jnp.isfinite(lam_cur), lam_cur, overflow_bound)
    prop_rej = jnp.maximum(prop_rej, jnp.maximum(lam_rej + shift_h + eps, 0.0))
    return lam_cur, prop_rej, _next_gamma(gamma_eff, prop_rej, eps, floor, cfg)


def _trial_loop(X, F, c, init, floor, eps, dt, n_steps, cfg, terms_fn, scan_fn, flt_max, res_ref=None, shift_h=0.0,
                jac_fn=None, lam_known=False):
    """Bounded loop over the damped trials after a rejected first trial. init is the first trial's carry (trial
    index 1, its gamma_eff, ok, X_new, F_new, c_new, the exponent of the current iterate or NaN, step_inf, the
    proposal). When the first trial was accepted the loop condition is false and init passes through, so the
    accepted outputs reach the caller without a lax.cond and without a copy of the Jacobians. jac_fn (the
    residual path, where F is not carried) recomputes the Jacobians of the current iterate for every retry, so
    that the first trial's F need not be kept live across the loop. lam_known (static): init's exponent is the
    exact one of the current iterate, so the retry body does not recompute it (see `_retry_gamma`)."""
    def body(carry):
        trial, gamma_prev, _, _, _, _, lam_cur, _, prop_c = carry
        Fb = F if jac_fn is None else jac_fn()               # Jacobians of the current iterate, recomputed on retry
        lam_cur, prop_c, gamma = _retry_gamma(gamma_prev, prop_c, lam_cur, Fb, floor, eps, dt, n_steps, cfg, flt_max,
                                              shift_h, lam_known)
        ok, X_new, F_new, c_new, step_inf, gamma_eff, prop_c = _trial(
            X, Fb, c, gamma, trial, prop_c, dt, cfg, terms_fn, scan_fn, res_ref)
        return (trial + 1, gamma_eff, ok, X_new, F_new, c_new, lam_cur, step_inf, prop_c)

    def cond(carry):
        trial, _, ok, _, _, _, _, _, _ = carry
        return (~ok) & (trial <= cfg.n_retry)

    return lax.while_loop(cond, body, init)


def run_damped_newton(X0: jnp.ndarray, terms_fn: Callable, scan_fn: Callable,
                      dt: float, cfg: LaserConfig, tol: float, max_iter: int,
                      residual_fn: Optional[Callable] = None):
    """LASER's damped Newton iteration.

    Args:
        X0: initial guess (N, D), excluding x0.
        terms_fn: X -> (F, c): step Jacobians F, shape (N-1, D, D),
            and c = -r(X), shape (N, D).
        scan_fn: (F_scaled, c) -> delta (N, D), the plain DEER scan on the scaled Jacobians.
        dt: step size (1.0 for discrete maps).
        cfg: LaserConfig.
        tol, max_iter: the loop runs while max|r| > tol, for at most max_iter iterations.
        residual_fn: optional X -> c, the residual alone, for callers whose Jacobians and residual are
            separate computations (explicit). When given and the exponent is not needed every
            iteration (plain-first without history), the loop carries (X, c) only: the Jacobians are
            formed by terms_fn at the top of each iteration (its residual part is pruned) and fused
            into the scan as in plain DEER, trials evaluate residual_fn, and a retry recomputes them.
            Otherwise (F, c) of the accepted trial are carried.

    Returns:
        X, residual max|r|, n_iterations, n_retries, history (dict of (max_iter,) arrays
        or None). history["lambda_h"][k] is the exact horizon LLE of the iterate produced
        by accepted step k (NaN in "plain" mode without history, where it is not needed;
        with cfg.exponent="carry" the carried exponent: the guess's, or the current iterate's
        exact one when a retry of step k computed it);
        history["gamma"][k] is the accepted damping. If no finite step could be produced
        (the Jacobi fallback itself is non-finite) the last accepted iterate is returned
        with residual = inf. The loop
        also stops when an accepted update is at rounding level (the dtype's residual
        floor), so a tol below that floor does not burn max_iter.

    Structure of one iteration (profiled on the L40S): the first trial is
    straight-line code and its outputs seed the retry while_loop, whose condition is false when the trial
    was accepted, so the accepted outputs are carried without a lax.cond around the loop and without a
    select after it (the loop stops after a failure, so only the returned X keeps the last accepted
    iterate). The damping of the next trial is computed inside the retry loop only, and the history is
    one (max_iter, 5) array updated by one kernel per iteration. The plain-first accepted path with
    residual_fn then costs what the plain DEER body costs.

    The Jacobians and the exact exponent of the current iterate are formed at the top of the iteration:
    with residual_fn the trials evaluate the residual alone,
    so a solve of K accepted iterations runs K Jacobian passes and K tree products, like plain DEER's K
    Jacobian passes, and nothing is formed at the final iterate. Forming the new iterate's Jacobians in the
    trial's terms_fn and its exponent at the end of the iteration would cost K + 1 of each, the last pair unused,
    for the same iterates bit for bit (tests/test_laser_reference.py). Without residual_fn the Jacobians are
    still formed by the trial and carried. With the history, the exponents of the final iterate are formed
    once after the loop.
    """
    N = X0.shape[0]
    n_steps = N - 1
    eps = 1.0 / (N * dt)
    flt_max = jnp.finfo(X0.dtype).max
    need_lambda = cfg.propose_first == "lle" or cfg.return_history
    carry_lambda = cfg.exponent == "carry"                   # lle by construction (LaserConfig)
    # exact_lambda: the exact exponent of every iterate, one tree product at the top of its iteration, over the
    # Jacobians formed there. It is the exponent the retry would recompute, so the retry's guarded recomputation is a
    # no-op and is dropped (lam_known); the carry mode needs it (its loop is seeded with NaN).
    exact_lambda = need_lambda and not carry_lambda
    lam_known = exact_lambda
    # The Jacobians are carried, formed by the trial at the new iterate, only for a caller whose terms_fn is the one
    # evaluation it has; with residual_fn they are formed at the top of the iteration and the trials evaluate the
    # residual alone (see the docstring).
    carry_F = residual_fn is None
    eval_fn = terms_fn if carry_F else (lambda Xn: (None, residual_fn(Xn)))
    # The guess's exponent seeds the carry mode (carried until a retry replaces it); the exact mode forms its own at
    # the top of every iteration.
    seed_lambda = carry_lambda
    if carry_F or seed_lambda:
        F0, c0 = terms_fn(X0)
    else:
        F0, c0 = None, residual_fn(X0)
    lam0 = initial_lle(F0, dt) if seed_lambda else jnp.array(jnp.nan, X0.dtype)
    fields = HISTORY_FIELDS + (EXTRA_HISTORY_FIELDS if cfg.return_history and cfg.history_extra else ())
    hist0 = jnp.zeros((max(max_iter, 1) if cfg.return_history else 1, len(fields)), X0.dtype)
    eps_mach = jnp.finfo(X0.dtype).eps
    # The history records, in the row of the iteration that produced an iterate, that iterate's exact exponent (exact
    # mode) and largest window exponent (history_extra): both are formed at the top of the next iteration, and for
    # the final iterate once after the loop, from the trial iterate the body carries for it (X_hist).
    late_cols = ([0] if exact_lambda else []) + ([fields.index("lambda_w")] if cfg.history_extra else [])
    record_final = cfg.return_history and bool(late_cols)

    def iterate_exponents(F):
        """The history's late columns of an iterate with Jacobians F, in the order of late_cols."""
        out = [initial_lle(F, dt)] if exact_lambda else []
        return out + ([window_exponents(F, dt, cfg.min_block)[0]] if cfg.history_extra else [])

    def fill_late(hist, row, vals, valid):
        for col, v in zip(late_cols, vals):
            hist = hist.at[row, col].set(jnp.where(valid, jnp.asarray(v, X0.dtype), hist[row, col]))
        return hist

    def body(carry):
        X, F, c, res, res_best, gamma_acc, lam_prop, contractive_count, k, n_retries, done, hist = carry[:12]
        if F is None:
            F, _ = terms_fn(X)                               # Jacobians of the current iterate (residual part pruned)
        if exact_lambda:
            lam_prop = initial_lle(F, dt)                    # the exact exponent of the current iterate
        # Consecutive contractive iterates (the floor's switch-off) with the current one counted, the guess excepted:
        # each iterate is counted once, at the top of the iteration that starts from it.
        contractive_count = jnp.where((lam_prop + eps < 0) & (k > 0), contractive_count + 1, 0)
        if record_final:                                     # this iterate's exponents complete the previous row
            hist = fill_late(hist, jnp.maximum(k - 1, 0), iterate_exponents(F), k > 0)
        floor = _floor(gamma_acc, contractive_count, cfg)
        # Trust radius: shift = log(max|r| / Delta) with Delta = trust_radius * max(1, max|X|); the horizon of length L
        # is damped only while lambda + shift / (L dt) + eps > 0, i.e. while it would amplify the residual past Delta.
        if cfg.trust_radius is not None:
            shift = jnp.log(jnp.max(jnp.abs(c)) / (cfg.trust_radius * jnp.maximum(1.0, jnp.max(jnp.abs(X)))))
        else:
            shift = jnp.array(0.0, X0.dtype)
        shift_h = shift / (n_steps * dt)
        prop = jnp.where(jnp.isnan(lam_prop), 0.0, jnp.maximum(lam_prop + shift_h + eps, 0.0))
        gamma0 = _first_gamma(prop, floor, cfg)
        res_ref = res_best if cfg.accept_ref == "best" else None
        # First trial as straight-line code: in plain-first mode its gamma is a constant zero, so the scaling
        # branch folds away and the accepted path costs the scan, the terms and the acceptance test.
        ok0, X1, F1, c1, step1, gamma_eff0, prop1 = _trial(
            X, F, c, gamma0, jnp.int32(0), prop, dt, cfg, eval_fn, scan_fn, res_ref)
        # Further trials only after a rejection; the loop passes the first trial's carry through when accepted.
        # lam_prop is the exact exponent of the current iterate when known ("lle" mode / history), else NaN. With
        # exponent="carry" lam_prop is the carried (possibly stale) exponent, so the loop is seeded with NaN and the
        # first rejection computes the exact one of the current iterate (`_retry_gamma`).
        lam_init = jnp.array(jnp.nan, X0.dtype) if carry_lambda else lam_prop
        init = (jnp.int32(1), gamma_eff0, ok0, X1, F1, c1, lam_init, step1, prop1)
        # The retries scale the current Jacobians again: kept live across the retry loop where the exponent needs
        # them anyway, recomputed per retry on the plain path so that the first trial's F is not held.
        jac_fn = None if (carry_F or need_lambda) else (lambda: terms_fn(X)[0])
        trial, gamma, ok, X_new, F_new, c_new, lam_cur, step_inf, _ = _trial_loop(
            X, F, c, init, floor, eps, dt, n_steps, cfg, eval_fn, scan_fn, flt_max, res_ref, shift_h, jac_fn,
            lam_known=lam_known)
        if cfg.n_retry == 0:      # Jacobi-only: a rejected sole trial reports a zero step
            step_inf = jnp.where(ok, step_inf, 0.0)
        # The exponent carried to the next iteration: the carry mode's (the one a retry computed, of the current
        # iterate, else the carried one), the exact mode's current one (the next iteration forms its own).
        if carry_lambda:
            lam_new = jnp.where(jnp.isnan(lam_cur), lam_prop, lam_cur)
        elif exact_lambda:
            lam_new = lam_prop
        else:
            lam_new = jnp.array(jnp.nan, X0.dtype)
        gamma_acc = jnp.where(jnp.isfinite(gamma), gamma, 0.0)
        res_new = jnp.where(ok, jnp.max(jnp.abs(c_new)), jnp.inf)
        # An accepted update at rounding level means the residual floor of the dtype is reached.
        stalled = ok & (step_inf <= 8.0 * eps_mach * jnp.maximum(1.0, jnp.max(jnp.abs(X_new))))
        if cfg.return_history:                              # static: no bookkeeping kernels otherwise
            nan = jnp.array(jnp.nan, X0.dtype)              # the late columns are filled from the next iterate
            vals = [nan if exact_lambda else lam_new, jnp.max(gamma), res_new, step_inf, trial.astype(X0.dtype)]
            if cfg.history_extra:                           # the first trial is ok0 / gamma_eff0 / c1 above
                vals += [jnp.max(gamma_eff0), ok0.astype(X0.dtype), jnp.max(jnp.abs(c1)) / jnp.max(jnp.abs(c)),
                         jnp.mean(gamma), nan]
            hist = hist.at[k].set(jnp.stack([jnp.asarray(v, X0.dtype) for v in vals]))
        # The loop stops after a failure (done), so the rejected (F, c) are carried as they are; only the
        # returned iterate keeps the last accepted X, by an (N, D) select rather than a copy of the Jacobians.
        X_out = jnp.where(ok, X_new, X)
        res_best = jnp.where(ok, jnp.minimum(res_best, res_new), res_best)
        out = (X_out, F_new, c_new, res_new, res_best, gamma_acc, lam_new,
               contractive_count, k + 1, n_retries + trial - 1, (~ok) | stalled, hist)
        if record_final:          # the iterate the history's last row describes: the last trial's (Jacobi-only:
            out = out + (X_new if cfg.n_retry > 0 else jnp.where(ok, X_new, X),)   # the current one if rejected)
        return out

    def cond(carry):
        res, k, done = carry[3], carry[8], carry[10]
        return (res > tol) & (k < max_iter) & (~done)

    res0 = jnp.max(jnp.abs(c0))
    init = (X0, F0 if carry_F else None, c0, res0, res0, jnp.zeros((), X0.dtype), lam0.astype(X0.dtype),
            jnp.int32(0), jnp.int32(0), jnp.int32(0), jnp.bool_(False), hist0)
    if record_final:
        init = init + (X0,)
    out = lax.while_loop(cond, body, init)
    X, res, k, n_retries, hist = out[0], out[3], out[8], out[9], out[11]
    if record_final:                        # the final iterate's exponents complete the last row (history only)
        hist = fill_late(hist, jnp.maximum(k - 1, 0), iterate_exponents(terms_fn(out[12])[0]), k > 0)
    history = {name: hist[:, i] for i, name in enumerate(fields)} if cfg.return_history else None
    return X, res, k, n_retries, history
