"""LASER on an ODE of your own: a template with the pieces a new problem needs.

The problem is a forced Van der Pol oscillator (mu = 3) with RK4, dt = 0.01, N = 2000 steps, float64, from the zero
guess (the unstable equilibrium of the unforced oscillator, repeated), where plain DEER returns a non-finite iterate.
Three uses:

  1. a forward solve of an ODE with an input u(t), and LASER's record of its iterations;
  2. a batch of initial states under jax.vmap;
  3. the gradient of a loss with respect to a parameter through the solve (implicit-function adjoint), compared
     with backpropagation through the sequential rollout.

    python examples/own_ode.py
"""
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from parode import LaserConfig, explicit_par_integrate, implicit_solve, seq_integrate  # noqa: E402


def make_ode(params):                               # params -> f; f is f(t, x) or f(t, x, u), with u = input_fn(t)
    return lambda t, x, u: jnp.array([x[1], params["mu"] * (1.0 - x[0] ** 2) * x[1] - x[0] + u])


def forcing(t):
    return 0.5 * jnp.sin(t)


params = {"mu": jnp.array(3.0)}
van_der_pol = make_ode(params)
x0, dt, N, tol, max_iter = jnp.array([2.0, 0.0]), 0.01, 2000, 1e-9, 200
guess = jnp.zeros((N, 2))                           # the states x_1 .. x_N; x_0 is fixed


def error(sol, x0):
    return float(jnp.max(jnp.abs(sol - seq_integrate(van_der_pol, x0, dt, N, "rk4", input_fn=forcing))))


def check(size, name):
    """The solvers do not raise when they stop above tol (at max_iter, on a non-finite iterate of plain DEER, or at
    the rounding level of the dtype when tol is below it): the caller compares their second output with tol. A NaN
    fails the comparison too."""
    if not bool(jnp.all(size <= tol)):
        print(f"             {name} did not converge to tol = {tol:g}")


# 1. Forward solve. Without laser= the call is plain DEER, whose second output is the size of its last update.
deer = jax.jit(lambda x0, guess: explicit_par_integrate(van_der_pol, x0, guess, dt, max_iter, "rk4",
                                                        input_fn=forcing, tol=tol))
sol, update, iterations = deer(x0, guess)
print(f"1. forward   DEER  iterations {int(iterations):3d}   last update {float(update):.2e}")
check(update, "DEER")

# return_history adds a dict of arrays of length max_iter (lambda_h, gamma, residual, step_inf, n_trials) as the last
# output, their first `iterations` entries filled; return_retries adds the number of rejected trial steps after the
# iteration count.
laser = jax.jit(lambda x0, guess: explicit_par_integrate(
    van_der_pol, x0, guess, dt, max_iter, "rk4", input_fn=forcing, tol=tol,
    laser=LaserConfig(return_history=True), return_retries=True))
sol, residual, iterations, retries, history = laser(x0, guess)      # sol (N + 1, 2) with x0; residual max|r|
print(f"             LASER iterations {int(iterations):3d}   rejected trials {int(retries)}   "
      f"residual {float(residual):.2e}   max |parallel - sequential| {error(sol, x0):.2e}")
check(residual, "LASER")
print("             damping gamma of the first five iterations:",
      " ".join(f"{float(g):.3f}" for g in history["gamma"][:5]))

# 2. A batch of initial states: vmap over x0 (and over the guess, if each solve has its own).
batch = jax.jit(jax.vmap(lambda x0: explicit_par_integrate(van_der_pol, x0, guess, dt, max_iter, "rk4",
                                                           input_fn=forcing, tol=tol, laser=LaserConfig())))
X0 = jnp.array([[2.0, 0.0], [0.5, 0.5], [-1.0, 2.0], [0.1, -0.1]])
sols, residuals, iterations = batch(X0)
print(f"2. batch     LASER iterations {[int(i) for i in iterations]}   max |parallel - sequential| "
      f"{max(error(s, x) for s, x in zip(sols, X0)):.2e}")
check(residuals, "a solve of the batch")


# 3. Gradient through the solve: implicit_solve takes make_ode and params and is differentiable in params and x0
# (and in an array inputs=, shape (N, d_in), in place of input_fn: step k reads inputs[k], the last stage of RK4
# inputs[k + 1]). Its one-step method is the keyword solver. A solve whose residual is above tol gets a zero gradient
# (unconverged="nan" or "pass" change that), so the loss returns the residual beside its value.
def loss_parallel(params):
    traj, residual, iterations = implicit_solve(make_ode, params, x0, guess, dt, solver="rk4", input_fn=forcing,
                                                tol=tol, max_iter=max_iter, laser=LaserConfig())
    return jnp.sum(traj[-1] ** 2), residual


def loss_sequential(params):
    return jnp.sum(seq_integrate(make_ode(params), x0, dt, N, "rk4", input_fn=forcing)[-1] ** 2)


grads, residual = jax.jit(jax.grad(loss_parallel, has_aux=True))(params)
g_parallel = float(grads["mu"])
g_sequential = float(jax.jit(jax.grad(loss_sequential))(params)["mu"])
print(f"3. gradient  d loss / d mu through the parallel solve {g_parallel:.6f}, "
      f"by backpropagation through the sequential rollout {g_sequential:.6f}")
check(residual, "the solve under the gradient")
if not abs(g_parallel - g_sequential) <= 1e-5 * abs(g_sequential):
    print("             the two gradients differ by more than 1e-5 of their size")
