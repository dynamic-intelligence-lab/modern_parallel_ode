"""Loss and accuracy for sequence classification.

Both functions handle scalar and batched inputs uniformly via mean reduction.
"""
from __future__ import annotations
from typing import Tuple

import jax
import jax.numpy as jnp


def cross_entropy_and_acc(
    logits: jnp.ndarray, labels: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Cross-entropy loss + top-1 accuracy.

    logits: (..., nclass), labels: (...,) int.
    Returns scalar (loss, acc) averaged over all leading dims.
    """
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(log_probs, labels[..., None], axis=-1).squeeze(-1)
    loss = nll.mean()
    pred = jnp.argmax(logits, axis=-1)
    acc = (pred == labels).mean()
    return loss, acc
