"""
Planar N-link robotic arm dynamics via Lagrangian mechanics.

Uses analytical mass matrix with JAX autodiff for Coriolis/gravity terms.
All dynamics follow the parode convention: ode(t, x, u=None) -> dx/dt

State convention: x = [q0, ..., q_{n-1}, qdot0, ..., qdot_{n-1}]
  q_i = relative joint angle (q=0 horizontal right, q=pi/2 up, q=-pi/2 down)
  Absolute angle of link i: alpha_i = sum(q[0:i+1])

ARM_2LINK is the arm of the neural-controller table (tab:controllers). The Slotine-Li rollout uses another two-link
arm, with a payload, defined in run_slotine_li.py; it does not use this module.
"""
import jax
import jax.numpy as jnp


def planar_arm(n_links, masses=None, lengths=None, gravity=9.81, damping=0.0):
    """
    Planar N-link arm with point masses at link endpoints.

    Returns:
        ode: function (t, x, u=None) -> dx/dt
        D: state dimension (2 * n_links)
        mass_matrix: function q -> M(q), shape (n, n)
        potential_energy: function q -> V(q), scalar
        fk_endpoints: function q -> (x_positions, y_positions), each (n,)
    """
    n = n_links
    D = 2 * n
    m = jnp.ones(n) if masses is None else jnp.asarray(masses, dtype=jnp.float64)
    l = jnp.ones(n) if lengths is None else jnp.asarray(lengths, dtype=jnp.float64)
    g = float(gravity)
    b = float(damping)

    idx = jnp.arange(n)
    # mask[k, i] = 1 if joint i affects mass k (i.e., i <= k)
    mask = (idx[:, None] >= idx[None, :]).astype(jnp.float64)
    sqrt_m = jnp.sqrt(m)

    def fk_endpoints(q):
        """Endpoint positions of each link. Returns (x, y) each shape (n,)."""
        abs_a = jnp.cumsum(q)
        return jnp.cumsum(l * jnp.cos(abs_a)), jnp.cumsum(l * jnp.sin(abs_a))

    def mass_matrix(q):
        """Analytical mass matrix M(q) via Jacobian outer products.

        M = sum_k m_k J_k^T J_k where J_k is the position Jacobian for mass k.
        """
        abs_a = jnp.cumsum(q)
        ls = l * jnp.sin(abs_a)
        lc = l * jnp.cos(abs_a)

        # Prefix sums (with leading zero) for suffix-sum computation
        cs_ls = jnp.concatenate([jnp.zeros(1), jnp.cumsum(ls)])
        cs_lc = jnp.concatenate([jnp.zeros(1), jnp.cumsum(lc)])

        # SS[k, i] = sum_{j=i}^{k} l_j sin(alpha_j) for i <= k, else 0
        SS = (cs_ls[idx + 1][:, None] - cs_ls[idx][None, :]) * mask
        SC = (cs_lc[idx + 1][:, None] - cs_lc[idx][None, :]) * mask

        # M = SS^T diag(m) SS + SC^T diag(m) SC
        wSS = sqrt_m[:, None] * SS
        wSC = sqrt_m[:, None] * SC
        M = wSS.T @ wSS + wSC.T @ wSC
        return M + 1e-12 * jnp.eye(n)  # regularize for numerical stability

    def potential_energy(q):
        """V = sum_k m_k g y_k."""
        _, y = fk_endpoints(q)
        return jnp.sum(m * g * y)

    def ode(t, x, u=None):
        q, qdot = x[:n], x[n:]

        M = mass_matrix(q)

        # dT/dq = (1/2) d/dq [qdot^T M(q) qdot]
        dT_dq = jax.grad(lambda q_: 0.5 * qdot @ mass_matrix(q_) @ qdot)(q)

        # dV/dq
        dV_dq = jax.grad(potential_energy)(q)

        # d/dt[M(q)*qdot] = Mdot*qdot via JVP
        _, Mdot_qdot = jax.jvp(
            lambda q_: mass_matrix(q_) @ qdot, (q,), (qdot,)
        )

        # Total torque (external + damping)
        tau = -b * qdot
        if u is not None:
            tau = tau + u

        # EOM: M*qddot = tau + dT/dq - dV/dq - Mdot*qdot
        # Use Cholesky (M is SPD) for better numerical stability on larger arms
        rhs = tau + dT_dq - dV_dq - Mdot_qdot
        L = jnp.linalg.cholesky(M)
        qddot = jax.scipy.linalg.solve_triangular(
            L.T, jax.scipy.linalg.solve_triangular(L, rhs, lower=True),
            lower=False
        )
        return jnp.concatenate([qdot, qddot])

    return ode, D, mass_matrix, potential_energy, fk_endpoints


# ============================================================
# The arm of the neural-controller table
# ============================================================

ARM_2LINK = dict(n_links=2, masses=[1.0, 1.0], lengths=[1.0, 1.0],
                 gravity=9.81, damping=0.1)
