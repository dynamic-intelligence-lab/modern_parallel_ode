# The solvers of `parode`

The short version is in the [README](../README.md#solve-your-own-ode).

```python
from parode import LaserConfig, explicit_par_integrate

sol, residual, iterations = explicit_par_integrate(f, x0, guess, dt, max_iter, "rk4", tol=tol, laser=LaserConfig())
```

| argument | meaning |
|---|---|
| `f` | the vector field, `f(t, x)`, or `f(t, x, u)` with the keyword `input_fn=` giving `u = input_fn(t)` at every evaluation time; `x` has shape (D,). `f` is called as `f(t, x)` when it has exactly two positional parameters and as `f(t, x, u)` otherwise (`u` is None without `input_fn=`), so a third positional parameter with a default receives `u`: make options keyword-only (`def f(t, x, *, mu=3.0)`) or close over them |
| `x0`, `guess` | the initial state, shape (D,), and a guess of the N states after it, shape (N, D); to warm-start from an earlier solution `sol` (N + 1 rows), pass `sol[1:]` |
| `dt`, `"euler"` or `"rk4"` | the step size and the one-step method; the grid is uniform |
| `max_iter`, `tol` | the cap on the Newton iterations and the tolerance on the residual max\|r\|; neither has a default, and `tol` is passed by keyword. A `tol` below the rounding level of the dtype is never reached; that level is relative to the size of the states (in float32, about 1e-7 times the largest state), so use 1e-5 to 1e-6 in float32 for states of order one (the paper's float32 ELCD runs use 1e-5). A float32 run needs `jax_enable_x64` off: with it on, float32 arrays fail in the solver's while loop with a type error of its carry |
| `laser=LaserConfig()` | LASER with the configuration of the paper's runs; without `laser=` the call is plain DEER, which stops on the size of its update and returns that size as its second output |

It returns the trajectory with x0, shape (N + 1, D), the residual and the iteration count. The call does not raise
when it stops above `tol`: at `max_iter`, on a non-finite iterate of plain DEER, or, for LASER, at the rounding level
of the dtype when `tol` is below it. Compare the second output with `tol` (`examples/own_ode.py` prints a line when it
is above `tol`).
`return_retries=True` adds the number of rejected trial steps, and `LaserConfig(return_history=True)` adds, as the
last output, a dict of arrays of length `max_iter` (`lambda_h`, `gamma`, `residual`, `step_inf`, `n_trials`) whose
first `iterations` entries describe the iterations. Wrap the call in `jax.jit`; `jax.vmap` gives a batch.
`seq_integrate(f, x0, dt, N, "rk4", input_fn=...)`, also from `parode`, is the sequential rollout, shape (N + 1, D).

`implicit_solve(make_ode, params, x0, guess, dt, solver="rk4", tol=tol, max_iter=max_iter, laser=LaserConfig())`,
also importable from `parode`, is the same solve of the ODE `make_ode(params)` with a gradient in `params`, `x0` and
`inputs` by the implicit-function theorem, for training through it. Its one-step method is the keyword `solver`
(`"euler"` when omitted), `tol` and `max_iter` are required keywords, its second output is the residual with and
without `laser=`, and it has no history.
`inputs` is an (N, d_in) array, which reaches `f` as a vector `u` of shape (d_in,): at a stage time t it is
`inputs[floor((t - t0)/dt + 1/4)]`, clipped to the array, so step k reads `inputs[k]`, except the last stage of RK4,
which reads `inputs[k + 1]`. `parode.implicit.seq_solve` is the sequential rollout with the same reading. `input_fn=`
works too and is not differentiated. A solve whose residual is above `tol` gets a zero gradient
(`unconverged="nan"` or `"pass"` changes that), and a `tol` below the rounding level is never reached: in float32
with too small a `tol` every gradient is zero and training does not move.

A recurrence that is not an explicit Euler or RK4 step calls LASER's loop itself, `parode.laser.run_damped_newton`,
with its own step Jacobians and scan (`tests/test_laser_carry.py` calls it with the three functions that
`euler_terms` in `tests/ode_problems.py` builds).

```sh
python examples/own_ode.py
```

runs the forward solve with the history and the rejected trials, a `jax.vmap` batch and the gradient in mu through
`implicit_solve` on a forced Van der Pol oscillator (mu = 3, forcing 0.5 sin t, RK4, 2000 steps of 0.01, float64,
zero guess), in 10 to 13 s on four CPU cores:

```
1. forward   DEER  iterations   2   last update nan
             DEER did not converge to tol = 1e-09
             LASER iterations  47   rejected trials 32   residual 5.15e-10   max |parallel - sequential| 4.99e-08
             damping gamma of the first five iterations: 2.683 9.183 9.183 9.183 9.183
2. batch     LASER iterations [47, 70, 74, 57]   max |parallel - sequential| 4.99e-08
3. gradient  d loss / d mu through the parallel solve 2.030847, by backpropagation through the sequential rollout 2.030847
```

Plain DEER is expected to fail from this guess, the unstable equilibrium of the unforced oscillator at every step.

The tolerance bounds the residual of the discretized equations, not the distance to the sequential solution: here
that distance is 5x10^-8 at a residual of 5x10^-10.
