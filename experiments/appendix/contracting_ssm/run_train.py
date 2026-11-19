"""One training run of the contracting SSM on EigenWorms. The defaults are the values of the paper's runs; the sweep
sets --gamma_min, the runs without the gamma head set --gamma_param off and larger iteration caps."""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from experiments.appendix.contracting_ssm.training import train  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(
        description="Train the nonlinear SSM on eigenworms.",
    )

    # ---- Solver ----
    p.add_argument("--solver", choices=["seq", "jacobi", "quasi"],
                   required=True,
                   help="seq: the sequential loop; jacobi: Jacobi sweeps; quasi: quasi-DEER. All three solve "
                        "the explicit-Euler recurrence (dt*(gamma_max + alpha_max) < 2 is needed).")
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--tol", type=float, default=1e-4,
                   help="jacobi and quasi stop once the largest residual of the Euler recurrence is below tol.")
    p.add_argument("--max_iter_jacobi", type=int, default=1000)
    p.add_argument("--max_iter_newton", type=int, default=200)

    # ---- Model ----
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--ffn_mult", type=int, default=2)
    p.add_argument("--init_alpha_range", type=str, default="0.1,1.5",
                   help="alpha_min,alpha_max. For A_param=exp: HiPPO init range. "
                        "For A_param=sigmoid: alpha_max is used as the upper bound; "
                        "the lower bound comes from --alpha_min.")
    p.add_argument("--A_param", choices=["exp", "sigmoid"], default="sigmoid",
                   help="A_ii parametrization. 'sigmoid' (default) = "
                        "-(alpha_min + (alpha_max - alpha_min) * sigmoid(g)), bounded in "
                        "[-alpha_max, -alpha_min], as Euler stability needs. 'exp' = -exp(-g): "
                        "log-uniform timescales but unbounded. Note: with A*tanh(h), alpha_min "
                        "alone does not force uniform strict contraction once h saturates.")
    p.add_argument("--alpha_min", type=float, default=0.0,
                   help="Lower bound on |A_ii| for sigmoid mode (0 in the paper's runs). "
                        "alpha_min > 0 bounds the unsaturated "
                        "Euler factor but does not prevent the tanh-saturated propagator "
                        "from approaching 1. "
                        "Trade-off: time constants capped at 1/alpha_min, shortening memory "
                        "horizon. Ignored when A_param=exp.")
    p.add_argument("--gamma_param", choices=["off", "sigmoid"], default="sigmoid",
                   help="Leak term parametrization. 'sigmoid' (default) adds -gamma(u)*h with "
                        "gamma in [gamma_min, gamma_max], giving a state-independent contraction "
                        "floor of gamma_min. Requires dt*(gamma_max + alpha_max) < 2 for Euler "
                        "stability. 'off' is the A*tanh cell without the leak term (the paper's "
                        "runs without the gamma head).")
    p.add_argument("--gamma_min", type=float, default=0.01,
                   help="Lower bound on the leak rate gamma(u); the paper sweeps 0.01, 0.02, "
                        "0.05, 0.1 and 0.2. Sets the contraction "
                        "floor: |G'(h)| <= 1 - dt*gamma_min. Smaller -> longer memory "
                        "(memory horizon ~1/gamma_min) but slower Newton convergence. "
                        "Ignored when gamma_param=off.")
    p.add_argument("--gamma_max", type=float, default=0.4,
                   help="Upper bound on the leak rate gamma(u). Combined with alpha_max "
                        "and dt, must satisfy dt*(gamma_max + alpha_max) < 2 for Euler "
                        "stability. Ignored when gamma_param=off.")

    # ---- Data ----
    p.add_argument("--decimate_stride", type=int, default=1,
                   help="Keep every n-th time step (1 in the paper's runs; 64 for the small-scale "
                        "runs of the README).")
    p.add_argument("--batch_size", type=int, default=8)

    # ---- Optimizer ----
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=30)

    # ---- Misc ----
    p.add_argument("--seed", type=int, default=0)

    # ---- Logging ----
    p.add_argument("--wandb_project", type=str, default=None,
                   help="wandb project; logging is off unless given (the paper runs used one).")
    p.add_argument("--wandb_group", type=str, default=None)

    return p.parse_args()


def main():
    args = parse_args()
    out = train(args)
    print("FINAL:", out)


if __name__ == "__main__":
    main()
