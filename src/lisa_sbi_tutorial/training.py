from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from .model import EmbeddingFlow


class TrainState(NamedTuple):
    model: EmbeddingFlow
    opt_state: optax.OptState


@eqx.filter_jit
def loss_fn(model: EmbeddingFlow, theta_batch: jax.Array, ts_batch: jax.Array):
    log_probs = jax.vmap(model.log_prob_physical)(theta_batch, ts_batch)
    return -jnp.mean(log_probs)


@eqx.filter_jit
def train_step(
    state: TrainState,
    theta_batch: jax.Array,
    ts_batch: jax.Array,
    optimizer: optax.GradientTransformation,
):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(
        state.model,
        theta_batch,
        ts_batch,
    )
    updates, new_opt_state = optimizer.update(
        grads,
        state.opt_state,
        eqx.filter(state.model, eqx.is_inexact_array),
    )
    new_model = eqx.apply_updates(state.model, updates)
    return TrainState(new_model, new_opt_state), loss


def train_online(
    key: jax.Array,
    model: EmbeddingFlow,
    batch_fn: Callable[[jax.Array, object | None, int], tuple[jax.Array, jax.Array]],
    *,
    experiment: object | None = None,
    learning_rate: float = 5e-4,
    batch_size: int = 256,
    max_epochs: int = 500,
    steps_per_epoch: int = 100,
    val_batch_fn: Callable[[jax.Array, object | None, int], tuple[jax.Array, jax.Array]] | None = None,
    val_steps: int = 1,
    patience: int = 20,
) -> tuple[EmbeddingFlow, dict[str, list[float]]]:
    """Train from batches generated on the fly.

    ``batch_fn(key, batch_size)`` must return standardized parameters and the
    corresponding time-domain observations.
    """
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    state = TrainState(model, opt_state)

    best_val = jnp.inf
    best_model = model
    patience_count = 0
    history = {"train": [], "val": []}
    val_batch_fn = batch_fn if val_batch_fn is None else val_batch_fn

    for epoch in range(max_epochs):
        train_losses = []
        for _ in range(steps_per_epoch):
            key, batch_key = jr.split(key)
            theta_batch, ts_batch = batch_fn(batch_key, experiment, batch_size)
            state, loss = train_step(state, theta_batch, ts_batch, optimizer)
            train_losses.append(loss)

        val_losses = []
        for _ in range(val_steps):
            key, val_key = jr.split(key)
            theta_val, ts_val = val_batch_fn(val_key, experiment, batch_size)
            val_losses.append(loss_fn(state.model, theta_val, ts_val))

        train_loss = jnp.mean(jnp.asarray(train_losses))
        val_loss = jnp.mean(jnp.asarray(val_losses))

        history["train"].append(float(train_loss))
        history["val"].append(float(val_loss))

        if val_loss < best_val:
            best_val = val_loss
            best_model = state.model
            patience_count = 0
        else:
            patience_count += 1

        if epoch % 20 == 0:
            print(f"Epoch {epoch:4d} | train {train_loss:.4f} | val {val_loss:.4f}")

        if patience_count >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    return best_model, history
