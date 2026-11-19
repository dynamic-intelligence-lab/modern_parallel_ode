"""How often a trained chain8z ELCD model is locally expanding in the Euclidean norm (the ELCD appendix of the paper).

  python experiments/elcd/train/sym_jacobian_positive.py fp32 laser

For each seed 0-9: the fraction of the training-demonstration samples x at which the symmetric part of the latent
Jacobian Df(phi(x)) has a positive eigenvalue, with z* = phi(x_eq) as in training (losses.Theta). Also asserts that
sym A(z) <= -alpha at every sample, so any positive eigenvalue comes from the term (dA/dz)(z - z*). CPU is enough.
Result (fp32 laser): 22.4, 45.0, 70.1, 68.2, 64.8, 78.2, 49.3, 84.6, 64.7, 61.6 % of 23067 samples.
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.elcd import flow_gatherfree as FG  # noqa: E402
from experiments.elcd import model as model_mod  # noqa: E402
from experiments.elcd.record_names import record_file  # noqa: E402
from experiments.elcd.train import data as D  # noqa: E402
from experiments.elcd.train.train_elcd import load_model  # noqa: E402

jax.config.update("jax_enable_x64", True)

ROOT = "results/elcd/records/main/chain8z"   # the chain8z records of the ELCD paper runs


def max_sym_eig(M):
    """Largest eigenvalue of the symmetric part of each (D, D) matrix in M (n, D, D)."""
    return np.asarray(jnp.linalg.eigvalsh(0.5 * (M + jnp.swapaxes(M, 1, 2)))[:, -1])


def main(dtype: str, solver: str):
    demos = D.load_system("chain8z")
    X = jnp.asarray(demos.pos[: demos.n_train].reshape(-1, demos.D))  # (3 N, D), normalized
    x_eq = jnp.asarray(demos.x_eq)
    print(f"{solver} {dtype}: {X.shape[0]} training samples, D = {demos.D}")
    for s in range(10):
        theta = load_model(record_file(f"{ROOT}/{dtype}/s{s}/{solver}.npz"))
        elcd = theta.elcd._replace(z_star=FG.phi(theta.flow, x_eq))
        Z = jax.vmap(lambda x: FG.phi(theta.flow, x))(X)
        lam = max_sym_eig(jax.vmap(lambda z: model_mod.jacobian(elcd, z))(Z))
        lam_A = max_sym_eig(jax.vmap(lambda z: model_mod.A_matrix(elcd, z))(Z))
        assert lam_A.max() <= -elcd.alpha + 1e-9, lam_A.max()
        print(f"seed {s}: positive at {100 * np.mean(lam > 0):.1f} % ({np.sum(lam > 0)} / {lam.size}), "
              f"max {lam.max():.3g}, median {np.median(lam):.3g}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
