"""The wandb run name of a training run."""
from __future__ import annotations
from typing import Any


def make_run_name(args: Any) -> str:
    """Build a wandb run name from training args (argparse Namespace or similar)."""
    return (
        f"eigenworms"
        f"_solver={args.solver}"
        f"_L={args.num_layers}"
        f"_D={args.hidden_dim}"
        f"_lr={args.lr}"
        f"_seed={args.seed}"
        f"_decimate={args.decimate_stride}"
    )
