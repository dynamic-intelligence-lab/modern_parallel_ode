
import jax.numpy as jnp


def lorenz_dynamics(t, state, u=None, sigma=10.0, rho=10.0, beta=8.0/3.0):
    """
    Lorenz system:
        dx/dt = sigma * (y - x)
        dy/dt = x * (rho - z) - y
        dz/dt = x * y - beta * z

    The defaults are the paper's system, sigma=10, rho=10, beta=8/3: bistable, the origin a saddle and the foci C+-
    stable (for 1 < rho < 24.74). run_benchmark.make_system passes all three. The classical chaotic system has
    rho=28.
    """
    x, y, z = state[0], state[1], state[2]
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return jnp.array([dx, dy, dz])
