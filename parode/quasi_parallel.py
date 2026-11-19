# Third-party origin: the helper functions combine_fc, par_scan, par_init and newton_step, and the
# while_loop driver of quasi_par_integrate, are derived from parode/explicit_parallel.py of parallel-ode by Casian
# Iacob, https://github.com/casiacob/parallel-ode, and modified for a diagonal Jacobian; euler_step and affine_terms
# are those of parode/explicit_parallel.py.
# What was taken and changed: THIRD_PARTY_NOTICES.md at the repository root, section "parallel-ode".
import jax.numpy as jnp
from jax import jvp, vmap, lax
from typing import Callable, Optional

from parode.explicit_parallel import affine_terms, euler_step
from parode.ode_utils import prepare_ode_function


def linear_terms(X, ode, dt, step, times, input_fn):
    '''
    Computes the Diagonal of the Jacobian EFFICIENTLY using JVP.
    Avoids instantiating the full (D, D) matrix.
    '''
    D = X.shape[1]
    
    # Define the transition for a single time step
    def transition_fn(x, t):
        return x + step(ode, x, t, dt, input_fn)
    
    # We want the diagonal: J_{ii} = d(f_i)/d(x_i)
    # This is equivalent to J * e_i (the i-th column) projected onto e_i.
    # We can compute J * v using jvp without building J.
    
    def get_diagonal_element(i, x, t):
        # Create basis vector e_i
        v = jnp.zeros_like(x).at[i].set(1.0)
        
        # Compute directional derivative (J * e_i)
        # This gives us the i-th column of the Jacobian
        _, col_i = jvp(lambda s: transition_fn(s, t), (x,), (v,))
        
        # Return the i-th element of that column (the diagonal entry)
        return col_i[i]

    # Vectorize over the dimensions (0 to D-1)
    # This effectively runs D forward passes, but uses O(D) memory instead of O(D^2)
    # Note: For very large D, this loop can be slow, but it respects the memory complexity.
    indices = jnp.arange(D)
    
    # We need to map this over all time steps
    def compute_diag_for_state(x, t):
        return vmap(get_diagonal_element, in_axes=(0, None, None))(indices, x, t)

    batch_diag = vmap(compute_diag_for_state)(X[:-1], times[1:-1])
    return batch_diag


def combine_fc(elem1, elem2):
    Fij, cij = elem1
    Fjk, cjk = elem2

    # OPTIMIZATION: Element-wise multiplication (*) for diagonal approximation
    # Fij and Fjk are now vectors (D,)
    Fik = Fjk * Fij  
    cik = Fjk * cij + cjk
    return Fik, cik


def par_scan(elems):
    return lax.associative_scan(vmap(combine_fc), elems, reverse=False)


def par_init(F, c, x0):
    # F is (T-1, D), c is (T-1, D). The first element's F never enters the c-part.
    tF0 = jnp.zeros((1, F.shape[1]))
    
    # tc0 = F[0] * x0 + c[0]
    # x0 is c[0] from newton_step (D,)
    # F[0] is (D,)
    tc0 = F[0] * x0 + c[0]
    tc0 = tc0.reshape(1, -1)
    
    tF = jnp.vstack((tF0, F[1:]))
    tc = jnp.vstack((tc0, c[1:]))
    elems = (tF, tc)
    return elems


def newton_step(X, x0, ode, dt, step, times, input_fn):
    c = affine_terms(X, x0, ode, dt, step, times, input_fn)
    F = linear_terms(X, ode, dt, step, times, input_fn)

    elems = par_init(F, c[1:], c[0])
    elems = par_scan(elems)
    X_N = jnp.vstack((c[0], elems[1]))
    norm_res = jnp.max(jnp.abs(X_N))
    norm_c = jnp.max(jnp.abs(c))
    return X_N, norm_res, norm_c


def quasi_par_integrate(
    ode: Callable,
    x0: jnp.ndarray,
    initial_guess: jnp.ndarray,
    dt: float,
    max_iter: int,
    t0: float = 0.0,
    input_fn: Optional[Callable] = None,
    *,
    tol: float,
):
    """Quasi-DEER (diagonal Jacobian) parallel integration with the explicit Euler step. Returns (trajectory
    (N+1, D) with x0, max|r| of the iterate the last Newton step started from, iterations): the loop stops on that
    residual, so when it stops on tol the returned trajectory is one Newton step past the first iterate with
    max|r| <= tol."""
    ode_fn = prepare_ode_function(ode)
    step = euler_step
    nb_steps = initial_guess.shape[0]
    times = t0 + dt * jnp.arange(nb_steps + 1)

    def while_body(val):
        X, _, iteration = val
        X_N, _, phys_err = newton_step(X, x0, ode_fn, dt, step, times, input_fn)
        X_new = X + X_N
        iteration += 1
        return X_new, phys_err, iteration

    def while_cond(val):
        _, err, iteration = val
        return (err > tol) & (iteration < max_iter)

    sol, residual, nb_iterations = lax.while_loop(
        while_cond, while_body, (initial_guess, jnp.array(1e3), 0)
    )
    return jnp.vstack((x0, sol)), residual, nb_iterations
