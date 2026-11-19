"""
Utilities of the robot-control experiment.

Contains:
- MLP initialization and forward pass
- The Lissajous reference trajectory
"""
import jax
import jax.numpy as jnp


# ============================================================
# MLP (pure JAX, pytree-compatible)
# ============================================================

def init_mlp(key, sizes):
    """Initialize MLP with He initialization. sizes = [input, h1, ..., output]."""
    params = []
    for i in range(len(sizes) - 1):
        key, subkey = jax.random.split(key)
        scale = jnp.sqrt(2.0 / sizes[i])
        w = scale * jax.random.normal(subkey, (sizes[i], sizes[i + 1]))
        b = jnp.zeros(sizes[i + 1])
        params.append((w, b))
    return params


def apply_mlp(params, x):
    """Forward pass: tanh hidden layers, linear output."""
    h = x
    for w, b in params[:-1]:
        h = jnp.tanh(h @ w + b)
    w, b = params[-1]
    return h @ w + b


# ============================================================
# Reference trajectory generators
# ============================================================

def make_lissajous_reference(omega1=2.0, omega2=None, amp1=0.5, amp2=0.4,
                              phase2=jnp.pi / 3, q_hang=None):
    """Lissajous reference trajectory for 2-link arm.

    q_ref(t) = q_hang + [amp1 * sin(omega1 * t), amp2 * sin(omega2 * t + phase2)]

    With irrational frequency ratio (omega2 = omega1 * sqrt(2) by default),
    the pattern never repeats, creating a demanding tracking task.

    Returns: (q_ref_fn, qdot_ref_fn, q_hang)
    """
    if q_hang is None:
        q_hang = jnp.array([-jnp.pi / 2, 0.0])
    if omega2 is None:
        omega2 = omega1 * jnp.sqrt(2.0)  # irrational ratio

    amps = jnp.array([amp1, amp2])
    omegas = jnp.array([omega1, omega2])
    phases = jnp.array([0.0, phase2])

    def q_ref_fn(t):
        return q_hang + amps * jnp.sin(omegas * t + phases)

    def qdot_ref_fn(t):
        return amps * omegas * jnp.cos(omegas * t + phases)

    return q_ref_fn, qdot_ref_fn, q_hang
