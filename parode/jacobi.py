# Third-party origin: adapted from deer_alg in src/deer.py of
# https://github.com/lindermanlab/predictability_enables_parallelization
#     Copyright (c) 2025 Linderman Lab
#     MIT License
# Licence text and list of changes: THIRD_PARTY_NOTICES.md at the repository root, section
# "predictability_enables_parallelization".
import jax
import jax.numpy as jnp
from jax import vmap, lax

def jacobi_alg(
    f,
    initial_state,
    states_guess,
    drivers,
    num_iters,
    merit_fxn, # Added explicit argument
    reset=True,
    tol=5e-4,
):
    """
    Jacobi iteration on the discrete map f(state, driver): each sweep sets every state of the new iterate to f of the
    previous state of the current one, until merit_fxn(f, initial_state, states, drivers) <= tol or num_iters sweeps.
    Returns seven values in the layout of the source's deer_alg: (None, final states (T, D), sweeps, is_nan, final
    merit, None, None).
    """

    @jax.jit
    def _step(carry, args):
        """
        One sweep, with the signature of a lax.scan body as in the source.
        Args:
            carry: tuple of (states, is_nan)
            args: None
        Returns ((new_states, is_nan), (new_states, nan_mask, merit of the old states)); the second pair is the
        per-sweep trace of the source's lax.scan mode, which this file does not have, and the loop drops it.
        """
        states, is_nan = carry
        fs = vmap(f)(states[:-1], drivers[1:])  # (T-1, D)
        b0 = f(initial_state, drivers[0])  # h1=f(h0, e1)
        new_states = jnp.concatenate([b0[jnp.newaxis, :], fs])  # (T, D)
        is_nan = jnp.logical_or(jnp.isnan(new_states).any(), is_nan)
        nan_mask = jnp.isnan(new_states)
        if reset:
            new_states = jnp.where(nan_mask, states_guess, new_states)
        mf_val = merit_fxn(f, initial_state, states, drivers)
        return (new_states, is_nan), (new_states, nan_mask, mf_val)

    def body_func_single(iter_inp):
        """
        Body func when we only use one T
        iter_inp: tuple of (iter_idx, states, err, is_nan, extra)
            iter_idx: int, the current iteration
            states: (T,D), the current states
            err: current value of merit function
            is_nan: bool, tracker of whether there has been a nan in the deer evaluation trace
            there is an extra input
        """
        iter_idx, states, _, is_nan_, _ = iter_inp
        step_output, _ = _step((states, is_nan_), None)
        new_states, is_nan = step_output
        # Evaluate merit on new_states so the err in carry matches the iterate
        # in carry. Without this, cond_func compared merit(states_{k-1}) > tol
        # but body had already advanced to states_k → "stop one step late".
        new_err = merit_fxn(f, initial_state, new_states, drivers)
        return iter_idx + 1, new_states, new_err, is_nan, None

    def cond_func(iter_inp):
        """
        iter_inp: tuple of (iter_idx, states, err)
            iter_idx: int, the current iteration
            states: (T,D), the current states
            err: current value of merit function
            is_nan: bool, tracker of whether there has been a nan in the deer evaluation trace
        """
        iter_idx, _, err, *_ = iter_inp
        return jnp.logical_and(iter_idx < num_iters, err > tol)

    newton_steps, final_state, final_err, is_nan, iters_below = jax.lax.while_loop(
        cond_func,
        body_func_single,
        (
            0,
            states_guess,
            merit_fxn(f, initial_state, states_guess, drivers),
            False,
            None,
        ),
    )
    all_states = None
    return (all_states, final_state, newton_steps, is_nan, final_err, None, None)