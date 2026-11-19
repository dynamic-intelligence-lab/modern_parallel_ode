"""Eigenworms data loader (reuses DEER repo pickles).

Each split is a list `[x, y]` of torch tensors:
    x: (N, T, ninp) float32
    y: (N,) int64

We load to numpy at startup, z-score per channel using train statistics,
optionally decimate by a stride along the time axis, then yield JAX arrays.
"""
from __future__ import annotations
import os
import pickle
from pathlib import Path
from typing import Iterator, Tuple

import jax.numpy as jnp
import jax.random as jr
import numpy as np


# The neuralrde_split pickles of the DEER repository (download_data.sh in this folder fetches them).
DEFAULT_SPLIT_DIR = Path(os.environ.get(
    "EIGENWORMS_DIR", Path(__file__).resolve().parents[3] / "data" / "eigenworms"))


def _to_numpy(x):
    """torch tensor or numpy array -> numpy array."""
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


def load_eigenworms(split_dir=None, decimate_stride: int = 1, dtype=jnp.float32):
    """Load all three splits, z-score with train statistics, decimate in time.

    Returns a dict with `train_x, train_y, val_x, val_y, test_x, test_y` as
    JAX arrays plus `ninp` and `nclass` ints.
    """
    split_dir = Path(split_dir) if split_dir else DEFAULT_SPLIT_DIR

    def load(name: str):
        with open(split_dir / f"eigenworms_{name}.pkl", "rb") as f:
            x, y = pickle.load(f)
        return _to_numpy(x), _to_numpy(y)

    train_x, train_y = load("train")
    val_x,   val_y   = load("val")
    test_x,  test_y  = load("test")

    # Per-channel z-score over (samples, time) using train stats only
    mean = train_x.mean(axis=(0, 1), keepdims=True)
    std  = train_x.std(axis=(0, 1), keepdims=True) + 1e-6

    def normalize(x):
        x = (x - mean) / std
        if decimate_stride > 1:
            x = x[:, ::decimate_stride, :]
        return jnp.asarray(x, dtype=dtype)

    return dict(
        train_x=normalize(train_x), train_y=jnp.asarray(train_y),
        val_x=normalize(val_x),     val_y=jnp.asarray(val_y),
        test_x=normalize(test_x),   test_y=jnp.asarray(test_y),
        ninp=int(train_x.shape[-1]),
        nclass=int(max(int(train_y.max()), int(val_y.max()), int(test_y.max())) + 1),
    )


def iterate_batches(
    x: jnp.ndarray,
    y: jnp.ndarray,
    batch_size: int,
    key,
    shuffle: bool = True,
    drop_last: bool = True,
) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
    """Yield (x_batch, y_batch) one at a time. Drops the last partial batch by default."""
    n = x.shape[0]
    perm = jr.permutation(key, n) if shuffle else jnp.arange(n)
    n_batches = n // batch_size if drop_last else (n + batch_size - 1) // batch_size
    for i in range(n_batches):
        idx = perm[i * batch_size : (i + 1) * batch_size]
        yield x[idx], y[idx]
