"""Training loop for the stacked nonlinear SSM on a sequence-classification task.

Usage: train(args) with the arguments of experiments/appendix/contracting_ssm/run_train.py.
"""
from __future__ import annotations
import functools
import time
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from experiments.appendix.contracting_ssm.eigenworms_data import load_eigenworms, iterate_batches
from experiments.appendix.contracting_ssm.cell_solvers import solve, solve_diagnostics
from experiments.appendix.contracting_ssm.stacked_model import init_model, encode, apply_layer_block, pool_and_readout
from experiments.appendix.contracting_ssm.loss import cross_entropy_and_acc
from experiments.appendix.contracting_ssm.naming import make_run_name


# ============================================================
# Forward pass
# ============================================================

def _forward_one(model, x_seq, solver_static):
    """Single-sample forward: x_seq (T, ninp) -> logits (nclass,)."""
    h_seq = encode(x_seq, model)
    for layer in model.layers:
        ode_solve_fn = lambda xs, cp: solve(xs, cp, solver_static)
        h_seq = apply_layer_block(h_seq, layer, ode_solve_fn)
    return pool_and_readout(h_seq, model)


def _forward_one_diag(model, x_seq, solver_static):
    """Same forward as `_forward_one`, but collects per-layer solver iters,
    residual, and one-step-Jacobian stats. Eval-time logging only."""
    h_seq = encode(x_seq, model)
    iters_pl, res_pl = [], []
    jmax_pl, jmean_pl, jp99_pl, jfrac_pl = [], [], [], []
    for layer in model.layers:
        def ode_solve_fn(xs, cp):
            h, it, re, jmax, jmean, jp99, jfrac = solve_diagnostics(
                xs, cp, solver_static,
            )
            iters_pl.append(it);   res_pl.append(re)
            jmax_pl.append(jmax);  jmean_pl.append(jmean)
            jp99_pl.append(jp99);  jfrac_pl.append(jfrac)
            return h
        h_seq = apply_layer_block(h_seq, layer, ode_solve_fn)
    logits = pool_and_readout(h_seq, model)
    iters    = jnp.stack(iters_pl).astype(jnp.float32)    # (L,)
    res      = jnp.stack(res_pl).astype(jnp.float32)
    jac_max  = jnp.stack(jmax_pl).astype(jnp.float32)
    jac_mean = jnp.stack(jmean_pl).astype(jnp.float32)
    jac_p99  = jnp.stack(jp99_pl).astype(jnp.float32)
    jac_frac = jnp.stack(jfrac_pl).astype(jnp.float32)
    return logits, iters, res, jac_max, jac_mean, jac_p99, jac_frac


def _loss_and_acc(model, x_batch, y_batch, solver_static):
    logits = jax.vmap(_forward_one, in_axes=(None, 0, None))(
        model, x_batch, solver_static,
    )
    return cross_entropy_and_acc(logits, y_batch)


def _solver_max_iter(solver_static):
    """Iter cap that determines whether a solver call counted as 'converged'
    (iterations below the cap). Single source of truth for converged_frac."""
    solver = solver_static[0]
    max_iter_jacobi = solver_static[3]
    max_iter_newton = solver_static[4]
    if solver == "seq":
        return 1                       # seq counts 1 iteration: converged_frac is 0
    if solver == "jacobi":
        return max_iter_jacobi
    return max_iter_newton             # quasi


def _loss_and_acc_diag(model, x_batch, y_batch, solver_static):
    """Eval forward + per-layer solver diagnostics, vmapped over batch.

    Returns 12-tuple aggregated over batch×layers:
        loss, acc,
        iters_p25, iters_median, iters_p75, iters_max,
        res_max, converged_frac,
        jac_max, jac_mean, jac_p99, jac_frac_above_1

    Newton/Jacobi linear convergence is governed by the one-step propagator
    magnitude on the converged trajectory. jac_frac_above_1 catches strict
    local expansion; equality at 1.0 is a non-strict boundary case and is
    reflected in jac_max/p99 instead.
    """
    logits, iters, res, jmax, jmean, jp99, jfrac = jax.vmap(
        _forward_one_diag, in_axes=(None, 0, None),
    )(model, x_batch, solver_static)              # all shapes: (B, L)
    loss, acc = cross_entropy_and_acc(logits, y_batch)

    iters_flat = iters.reshape(-1)
    converged_frac = (iters_flat < _solver_max_iter(solver_static)).mean()
    qs = jnp.percentile(iters_flat, jnp.array([25.0, 50.0, 75.0]))
    return (loss, acc,
            qs[0], qs[1], qs[2], iters_flat.max(),
            res.max(), converged_frac,
            jmax.max(), jmean.mean(), jp99.max(), jfrac.mean())


# ============================================================
# Training loop
# ============================================================

def train(args: Any):
    key = jr.PRNGKey(args.seed)

    # ---- Data ----
    data = load_eigenworms(decimate_stride=args.decimate_stride)

    print(f"data: train={data['train_x'].shape} val={data['val_x'].shape} "
          f"test={data['test_x'].shape} ninp={data['ninp']} nclass={data['nclass']}")

    # ---- Model ----
    key, mkey = jr.split(key)
    init_alpha_range = tuple(float(a) for a in args.init_alpha_range.split(","))
    model = init_model(
        mkey,
        ninp=data["ninp"],
        D=args.hidden_dim,
        num_layers=args.num_layers,
        nclass=data["nclass"],
        ffn_mult=args.ffn_mult,
        init_alpha_range=init_alpha_range,
        A_param=args.A_param,
        gamma_param=args.gamma_param,
    )
    A_alpha_max = float(init_alpha_range[1])

    # ---- Optimizer ----
    n_train = data["train_x"].shape[0]
    steps_per_epoch = max(1, n_train // args.batch_size)
    total_steps = max(args.epochs * steps_per_epoch, 1)
    warmup = min(args.warmup_steps, max(total_steps // 4, 1))

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.lr,
        warmup_steps=warmup,
        decay_steps=total_steps,
        end_value=0.0,
    )
    opt = optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        optax.adamw(learning_rate=schedule, weight_decay=args.weight_decay),
    )
    opt_state = opt.init(model)

    solver_static = (
        args.solver, args.dt, args.tol,
        args.max_iter_jacobi, args.max_iter_newton,
        args.A_param, A_alpha_max, args.alpha_min,
        args.gamma_param, args.gamma_min, args.gamma_max,
    )

    # ---- JIT'd train/eval steps ----
    @functools.partial(jax.jit, static_argnames=("solver_static",))
    def train_step(model, opt_state, x_batch, y_batch, *, solver_static):
        def loss_only(m):
            return _loss_and_acc(m, x_batch, y_batch, solver_static)
        (loss, acc), grads = jax.value_and_grad(loss_only, has_aux=True)(model)
        updates, opt_state = opt.update(grads, opt_state, model)
        model = optax.apply_updates(model, updates)
        return model, opt_state, loss, acc

    @functools.partial(jax.jit, static_argnames=("solver_static",))
    def eval_step(model, x_batch, y_batch, *, solver_static):
        return _loss_and_acc_diag(model, x_batch, y_batch, solver_static)

    # ---- wandb (optional) ----
    use_wandb = bool(args.wandb_project)
    wandb = None
    if use_wandb:
        import wandb as _wandb
        wandb = _wandb
        wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=make_run_name(args),
            config=vars(args),
        )

    # ---- Loop ----
    best_val_acc = -1.0
    best_model = model
    patience_left = args.patience

    first_step_time_s = None  # captured once, on the very first train step
    epoch_wall_times: list[float] = []   # per-epoch elapsed seconds

    for epoch in range(args.epochs):
        t_epoch = time.perf_counter()
        key, ekey = jr.split(key)

        train_losses, train_accs, step_times_ms = [], [], []
        for xb, yb in iterate_batches(data["train_x"], data["train_y"],
                                      args.batch_size, ekey):
            t_step = time.perf_counter()
            model, opt_state, loss, acc = train_step(
                model, opt_state, xb, yb,
                solver_static=solver_static,
            )
            loss.block_until_ready()  # force async dispatch to complete for timing
            step_times_ms.append((time.perf_counter() - t_step) * 1000.0)
            train_losses.append(float(loss))
            train_accs.append(float(acc))

        if first_step_time_s is None:
            first_step_time_s = step_times_ms[0] / 1000.0

        val_losses, val_accs = [], []
        val_p25, val_p50, val_p75, val_iters_maxes = [], [], [], []
        val_res_maxes, val_conv_fracs = [], []
        val_jmax, val_jmean, val_jp99, val_jfrac = [], [], [], []
        for xb, yb in iterate_batches(data["val_x"], data["val_y"],
                                      args.batch_size, jr.PRNGKey(0),
                                      shuffle=False, drop_last=False):
            (vl, va, vi_p25, vi_p50, vi_p75, vi_max,
             vr_max, vc_frac,
             vj_max, vj_mean, vj_p99, vj_frac) = eval_step(
                model, xb, yb, solver_static=solver_static,
            )
            val_losses.append(float(vl));    val_accs.append(float(va))
            val_p25.append(float(vi_p25));   val_p50.append(float(vi_p50))
            val_p75.append(float(vi_p75));   val_iters_maxes.append(float(vi_max))
            val_res_maxes.append(float(vr_max)); val_conv_fracs.append(float(vc_frac))
            val_jmax.append(float(vj_max));   val_jmean.append(float(vj_mean))
            val_jp99.append(float(vj_p99));   val_jfrac.append(float(vj_frac))

        train_loss = sum(train_losses) / len(train_losses)
        train_acc  = sum(train_accs)   / len(train_accs)
        val_loss   = sum(val_losses)   / len(val_losses)
        val_acc    = sum(val_accs)     / len(val_accs)
        # Quantiles aggregated as means across batches; max-of-max for the worst
        # case; converged_frac as a mean across batches (fraction of solver
        # calls below max_iter, averaged over the val set).
        val_iters_p25  = sum(val_p25) / len(val_p25)
        val_iters_p50  = sum(val_p50) / len(val_p50)
        val_iters_p75  = sum(val_p75) / len(val_p75)
        val_iters_max  = max(val_iters_maxes)
        val_res_max    = max(val_res_maxes)
        val_conv_frac  = sum(val_conv_fracs) / len(val_conv_fracs)
        # Jacobian aggregations: max-of-batch-max (worst seen anywhere),
        # mean-of-mean (typical), max-of-p99 (robust worst), mean-of-frac.
        val_jac_max    = max(val_jmax)
        val_jac_mean   = sum(val_jmean) / len(val_jmean)
        val_jac_p99    = max(val_jp99)
        val_jac_frac   = sum(val_jfrac) / len(val_jfrac)
        elapsed    = time.perf_counter() - t_epoch
        epoch_wall_times.append(elapsed)

        # Step time excluding first step in *first* epoch (which includes JIT
        # compilation). On later epochs all steps are steady-state.
        steady_steps = step_times_ms[1:] if epoch == 0 and len(step_times_ms) > 1 else step_times_ms
        step_ms_mean = sum(steady_steps) / max(len(steady_steps), 1)

        log = {
            "epoch": epoch,
            "train/loss": train_loss, "train/acc": train_acc,
            "val/loss":   val_loss,   "val/acc":   val_acc,
            "wall_time_per_epoch_s": elapsed,
            "train/step_time_mean_ms": step_ms_mean,
            "train/first_step_time_s": first_step_time_s,
            "lr": float(schedule(epoch * steps_per_epoch)),
            "solver/iters_p25":      val_iters_p25,
            "solver/iters_median":   val_iters_p50,
            "solver/iters_p75":      val_iters_p75,
            "solver/iters_max":      val_iters_max,
            "solver/residual_max":   val_res_max,
            "solver/converged_frac": val_conv_frac,
            "solver/jac_max":         val_jac_max,
            "solver/jac_mean":        val_jac_mean,
            "solver/jac_p99":         val_jac_p99,
            "solver/jac_frac_above_1": val_jac_frac,
        }
        if use_wandb:
            wandb.log(log)
        print(f"epoch {epoch:3d}: train_loss={train_loss:.4f} acc={train_acc:.3f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.3f} | "
              f"iters={val_iters_p25:.0f}/{val_iters_p50:.0f}/{val_iters_p75:.0f}/{val_iters_max:.0f} "
              f"conv={val_conv_frac:.2f} res={val_res_max:.1e} "
              f"jac={val_jac_mean:.3f}/{val_jac_p99:.3f}/{val_jac_max:.3f} "
              f"jac>1={val_jac_frac:.2f} | "
              f"{elapsed:.1f}s ({step_ms_mean:.0f}ms/step)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"early stop at epoch {epoch}, best val_acc={best_val_acc:.3f}")
                break

    # ---- Wall-time summary ----
    # epoch 0 includes JIT compile (forward + backward + eval); subsequent
    # epochs are steady-state. Report both totals so we can compare solvers
    # at their actual training cost, separately from one-time compile overhead.
    total_wall_time_s = sum(epoch_wall_times)
    if len(epoch_wall_times) >= 2:
        train_wall_time_excl_jit_s = sum(epoch_wall_times[1:])
        steady_per_epoch = train_wall_time_excl_jit_s / (len(epoch_wall_times) - 1)
        jit_overhead_s_est = max(epoch_wall_times[0] - steady_per_epoch, 0.0)
    else:
        train_wall_time_excl_jit_s = 0.0
        jit_overhead_s_est = epoch_wall_times[0] if epoch_wall_times else 0.0
    print(f"WALL TIME: total={total_wall_time_s:.1f}s "
          f"train_excl_jit={train_wall_time_excl_jit_s:.1f}s "
          f"jit_overhead≈{jit_overhead_s_est:.1f}s "
          f"({len(epoch_wall_times)} epochs)")
    if use_wandb:
        wandb.log({
            "wall_time/total_s":               total_wall_time_s,
            "wall_time/train_excl_jit_s":      train_wall_time_excl_jit_s,
            "wall_time/jit_overhead_s_est":    jit_overhead_s_est,
            "wall_time/epochs_completed":      len(epoch_wall_times),
        })

    # ---- Test on best-val checkpoint ----
    test_losses, test_accs = [], []
    test_p50, test_iters_maxes, test_res_maxes, test_conv_fracs = [], [], [], []
    test_jmax, test_jmean = [], []
    for xb, yb in iterate_batches(data["test_x"], data["test_y"],
                                  args.batch_size, jr.PRNGKey(0),
                                  shuffle=False, drop_last=False):
        (tl, ta, _, ti_p50, _, ti_max, tr_max, tc_frac,
         tj_max, tj_mean, _, _) = eval_step(
            best_model, xb, yb, solver_static=solver_static,
        )
        test_losses.append(float(tl));   test_accs.append(float(ta))
        test_p50.append(float(ti_p50));  test_iters_maxes.append(float(ti_max))
        test_res_maxes.append(float(tr_max)); test_conv_fracs.append(float(tc_frac))
        test_jmax.append(float(tj_max)); test_jmean.append(float(tj_mean))
    test_loss          = sum(test_losses) / len(test_losses)
    test_acc           = sum(test_accs)   / len(test_accs)
    test_iters_median  = sum(test_p50)    / len(test_p50)
    test_iters_max     = max(test_iters_maxes)
    test_res_max       = max(test_res_maxes)
    test_conv_frac     = sum(test_conv_fracs) / len(test_conv_fracs)
    test_jac_max       = max(test_jmax)
    test_jac_mean      = sum(test_jmean) / len(test_jmean)
    print(f"TEST (best val): loss={test_loss:.4f} acc={test_acc:.3f} "
          f"iters median={test_iters_median:.0f} max={test_iters_max:.0f} "
          f"conv={test_conv_frac:.2f} res={test_res_max:.1e} "
          f"jac mean={test_jac_mean:.3f} max={test_jac_max:.3f}")

    if use_wandb:
        wandb.log({"test/loss": test_loss, "test/acc": test_acc,
                   "test/iters_median":   test_iters_median,
                   "test/iters_max":      test_iters_max,
                   "test/residual_max":   test_res_max,
                   "test/converged_frac": test_conv_frac,
                   "test/jac_max":        test_jac_max,
                   "test/jac_mean":       test_jac_mean,
                   "best_val_acc": best_val_acc})
        wandb.finish()

    return dict(
        best_val_acc=best_val_acc, test_acc=test_acc, test_loss=test_loss,
    )

