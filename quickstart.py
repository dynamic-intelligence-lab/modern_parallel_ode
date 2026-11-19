"""Quick start: the bistable Lorenz problem of the paper, on CPU in a few seconds.

Lorenz system with sigma = 10, rho = 10, beta = 8/3 from x0 = (1, 1, 1): explicit Euler, dt = 0.01, N = 1000 steps,
float64. Every solver starts from the zero guess, the constant trajectory at the saddle, where the step Jacobians
are unstable. Plain DEER (Newton's method with a parallel scan) returns a non-finite iterate; LASER damps the
Jacobians with the finite-time Lyapunov exponent of the current iterate and reaches the tolerance in 22 iterations.

    python quickstart.py
"""
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from parode import LaserConfig, explicit_par_integrate, seq_integrate  # noqa: E402


def lorenz(t, x, u=None, sigma=10.0, rho=10.0, beta=8.0 / 3.0):
    return jnp.array([sigma * (x[1] - x[0]), x[0] * (rho - x[2]) - x[1], x[0] * x[1] - beta * x[2]])


x0, dt, N, tol, max_iter = jnp.array([1.0, 1.0, 1.0]), 0.01, 1000, 1e-9, 200
guess = jnp.zeros((N, 3))                                   # the states x_1 .. x_N; x_0 is fixed
sequential = seq_integrate(lorenz, x0, dt, N, "euler")      # (N + 1, 3)

laser = LaserConfig()                                       # the defaults: the configuration of the paper's runs
for name, cfg in (("DEER", None), ("LASER", laser)):
    solve = jax.jit(lambda x0, guess: explicit_par_integrate(lorenz, x0, guess, dt, max_iter, "euler", tol=tol,
                                                             laser=cfg))
    # sol (N + 1, 3) with x0; the size the loop stops on (LASER: the residual max|r| of sol; plain DEER: the
    # size of its last update); the iteration count
    sol, residual, iterations = solve(x0, guess)
    error = float(jnp.max(jnp.abs(sol - sequential)))
    size = "last update" if cfg is None else "residual"
    print(f"{name:5s} iterations {int(iterations):3d}   {size} {float(residual):.2e}   "
          f"max |parallel - sequential| {error:.2e}")
    if name == "LASER":
        assert int(iterations) == 22 and error < 1e-9, "LASER did not reproduce the paper's run"
