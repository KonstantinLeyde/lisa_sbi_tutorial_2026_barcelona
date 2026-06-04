from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from flowjax.bijections import Affine, RationalQuadraticSpline
from flowjax.distributions import Normal
from flowjax.flows import coupling_flow


@dataclass(frozen=True)
class ModelConfig:
    ts_length: int
    embedding_dim: int
    param_dim: int
    mlp_width: int = 64
    mlp_depth: int = 3
    flow_layers: int = 8
    nn_width: int = 64
    nn_depth: int = 2


class EmbeddingFlow(eqx.Module):
    """MLP summary network plus conditional normalizing flow."""

    embedding: eqx.nn.MLP
    flow: eqx.Module
    config: ModelConfig = eqx.field(static=True)
    param_mean: jax.Array | None = None
    param_std: jax.Array | None = None
    param_names: tuple[str, ...] = eqx.field(default=(), static=True)

    @property
    def has_standardization(self) -> bool:
        return self.param_mean is not None and self.param_std is not None

    def with_standardization(
        self,
        mean: jax.Array,
        std: jax.Array,
        names: Sequence[str],
    ) -> "EmbeddingFlow":
        return EmbeddingFlow(
            embedding=self.embedding,
            flow=self.flow,
            config=self.config,
            param_mean=jnp.asarray(mean),
            param_std=jnp.asarray(std),
            param_names=tuple(names),
        )

    def standardize(self, theta: jax.Array) -> jax.Array:
        if not self.has_standardization:
            raise ValueError("Parameter standardization stats are not attached.")
        return (theta - self.param_mean) / self.param_std

    def unstandardize(self, theta_standardized: jax.Array) -> jax.Array:
        if not self.has_standardization:
            raise ValueError("Parameter standardization stats are not attached.")
        return theta_standardized * self.param_std + self.param_mean

    def log_prob(self, theta: jax.Array, condition: jax.Array) -> jax.Array:
        """Log p(theta_standardized | condition)."""
        condition = self.embedding(condition)
        return self.flow.log_prob(theta, condition)

    def log_prob_physical(self, theta: jax.Array, condition: jax.Array) -> jax.Array:
        """Log probability density for unstandardized physical parameters."""
        if not self.has_standardization:
            raise ValueError("Parameter standardization stats are not attached.")

        theta_standardized = self.standardize(theta)
        log_abs_det = jnp.sum(jnp.log(self.param_std))
        return self.log_prob(theta_standardized, condition) - log_abs_det

    def sample(
        self,
        key: jax.Array,
        condition: jax.Array,
        n_samples: int = 1000,
        *,
        standardized: bool = False,
    ) -> jax.Array:
        condition = self.embedding(condition)
        samples = self.flow.sample(key, (n_samples,), condition=condition)
        if standardized or not self.has_standardization:
            return samples
        return self.unstandardize(samples)

    def sample_standardized(
        self,
        key: jax.Array,
        condition: jax.Array,
        n_samples: int = 1000,
    ) -> jax.Array:
        return self.sample(key, condition, n_samples, standardized=True)


def make_model(
    key: jax.Array,
    *,
    ts_length: int,
    embedding_dim: int,
    param_dim: int,
    mlp_width: int = 64,
    mlp_depth: int = 3,
    flow_layers: int = 8,
    nn_width: int = 64,
    nn_depth: int = 2,
    param_mean: jax.Array | None = None,
    param_std: jax.Array | None = None,
    param_names: Sequence[str] = (),
    transformer: eqx.Module = Affine(),
) -> EmbeddingFlow:
    _, embedding_key, flow_key = jr.split(key, 3)
    config = ModelConfig(
        ts_length=ts_length,
        embedding_dim=embedding_dim,
        param_dim=param_dim,
        mlp_width=mlp_width,
        mlp_depth=mlp_depth,
        flow_layers=flow_layers,
        nn_width=nn_width,
        nn_depth=nn_depth,
    )

    embedding = eqx.nn.MLP(
        in_size=config.ts_length,
        out_size=config.embedding_dim,
        width_size=config.mlp_width,
        depth=config.mlp_depth,
        key=embedding_key,
    )

    flow = coupling_flow(
        flow_key,
        base_dist=Normal(jnp.zeros(config.param_dim)),
        transformer=transformer,
        cond_dim=config.embedding_dim,
        flow_layers=config.flow_layers,
        nn_width=config.nn_width,
        nn_depth=config.nn_depth,
    )

    model = EmbeddingFlow(
        embedding=embedding,
        flow=flow,
        config=config,
        param_names=tuple(param_names),
    )
    if param_mean is not None and param_std is not None:
        model = model.with_standardization(param_mean, param_std, param_names)
    return model


def save_model(path: str | Path, model: EmbeddingFlow) -> Path:
    """Save a model directory containing weights and standardization metadata."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    eqx.tree_serialise_leaves(path / "model.eqx", model)
    metadata = {
        "config": asdict(model.config),
        "param_names": list(model.param_names),
        "param_mean": None
        if model.param_mean is None
        else jnp.asarray(model.param_mean).tolist(),
        "param_std": None
        if model.param_std is None
        else jnp.asarray(model.param_std).tolist(),
    }
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return path


def load_model(path: str | Path, key: jax.Array | None = None) -> EmbeddingFlow:
    path = Path(path)
    metadata = json.loads((path / "metadata.json").read_text())
    config = ModelConfig(**metadata["config"])
    key = jr.key(0) if key is None else key

    param_mean_raw = metadata.get("param_mean")
    param_std_raw = metadata.get("param_std")

    has_standardization = param_mean_raw is not None and param_std_raw is not None

    model_template = make_model(
        key,
        **asdict(config),
        param_mean=None if not has_standardization else jnp.asarray(param_mean_raw),
        param_std=None if not has_standardization else jnp.asarray(param_std_raw),
        param_names=metadata.get("param_names", ()),
    )

    return eqx.tree_deserialise_leaves(path / "model.eqx", model_template)