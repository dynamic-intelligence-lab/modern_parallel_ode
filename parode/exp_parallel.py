# Third-party origin: newton_step is derived from parode/explicit_parallel.py of parallel-ode by Casian Iacob,
# https://github.com/casiacob/parallel-ode; the scan helpers it uses are those of parode/explicit_parallel.py.
# What was taken and changed: THIRD_PARTY_NOTICES.md at the repository root, section "parallel-ode".
import jax
import jax.numpy as jnp
import jax.scipy.linalg
from jax import vmap, jacfwd

from parode.explicit_parallel import par_init, par_scan
from parode.ode_utils import evaluate_input


def _compute_Ab(ode, x, t, input_fn):
    """Compute linearization A = df/dx and forcing b = f - A*x at (t, x).

    Returns:
        A: Jacobian, shape (D, D)
        b: forcing vector, shape (D,)
    """
    u = evaluate_input(input_fn, t)
    f_val = ode(t, x, u)
    A = jacfwd(lambda z: ode(t, z, u))(x)
    b = f_val - A @ x
    return A, b


def _augmented_expm(A, b, dt):
    """Compute Phi = expm(hA) and h*phi_1(hA)*b via the block-matrix trick.

    Returns:
        Phi: expm(h*A), shape (D, D)
        phi_b: h*phi_1(hA)*b, shape (D,)
    """
    D = A.shape[0]
    M = jnp.zeros((D + 1, D + 1))
    M = M.at[:D, :D].set(dt * A)
    M = M.at[:D, D].set(dt * b)
    E = jax.scipy.linalg.expm(M)
    return E[:D, :D], E[:D, D]


def _exp_step_and_phi(ode, x, t, dt, input_fn):
    """
    Compute the exponential integrator step and the recurrence coefficient
    in a single expm call using the block-matrix trick.

    For dx/dt = f(x) linearized as dx/dt = A*x + b:
        expm([[hA, hb], [0, 0]])  gives  [[expm(hA), h*phi_1(hA)*b], [0, 1]]

    Returns:
        Phi: expm(h*A), shape (D, D) -- recurrence coefficient
        x_next: expm(hA)*x + h*phi_1(hA)*b, shape (D,) -- one-step result
    """
    A, b = _compute_Ab(ode, x, t, input_fn)
    Phi, phi_b = _augmented_expm(A, b, dt)
    x_next = Phi @ x + phi_b
    return Phi, x_next


def linear_and_affine_terms(X, x0, ode, dt, times, input_fn):
    """
    Fused computation of recurrence coefficients and residual. The forcing b is frozen at the left endpoint
    of each step (zero-order hold), which uses phi_1.

    Returns:
        F: expm(h*A_n) for steps 1..N-1, shape (N-1, D, D)
        c: residual -X + exp_step(X_prev), shape (N, D)
    """
    # compute (Phi, x_next) at N prev points
    X_prev = jnp.vstack((x0, X[:-1]))
    t_prev = times[:-1]

    def single_step(x_prev, t_i):
        Phi, x_next = _exp_step_and_phi(ode, x_prev, t_i, dt, input_fn)
        return Phi, x_next

    Phis, x_nexts = vmap(single_step)(X_prev, t_prev)

    c = -X + x_nexts
    F = Phis[1:]
    return F, c


def newton_step(X, x0, ode, dt, times, input_fn):
    F, c = linear_and_affine_terms(X, x0, ode, dt, times, input_fn)
    elems = par_init(F, c[1:], c[0])
    elems = par_scan(elems)
    X_N = jnp.vstack((c[0], elems[1]))
    norm_res = jnp.max(jnp.abs(X_N))
    norm_c = jnp.max(jnp.abs(c))
    return X_N, norm_res, norm_c
