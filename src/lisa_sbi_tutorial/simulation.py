from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np


DEFAULT_PRIORS = {
    "amplitude": {"type": "uniform", "min": 0.2, "max": 2.0},
    "frequency": {"type": "uniform", "min": 5.0, "max": 40.0},
    "phase": {"type": "uniform", "min": 0.0, "max": 2 * jnp.pi},
    "damping_time": {"type": "uniform", "min": 0.2, "max": 1.0},
}


def make_time_grid(
    duration: float = 1.0,
    sample_rate: float = 512.0,
) -> jax.Array:
    num_samples = int(duration * sample_rate)
    return jnp.arange(num_samples) / sample_rate


def get_signal(
    times: jax.Array,
    params: Mapping[str, jax.Array | float],
) -> jax.Array:
    """Damped sinusoid signal model for one time-domain sample.

    Parameters may be Python scalars or JAX arrays that broadcast against
    ``times``.
    """
    envelope = jnp.exp(-times / params["damping_time"])
    phase = 2 * jnp.pi * params["frequency"] * times + params["phase"]
    return params["amplitude"] * envelope * jnp.sin(phase)


def get_signals(
    times: jax.Array,
    params: Mapping[str, jax.Array],
) -> jax.Array:
    times_batched = times[None, :]
    params_batched = {key: value[:, None] for key, value in params.items()}
    return get_signal(times_batched, params_batched)


def draw_noise(
    key: jax.Array,
    times: jax.Array,
    *,
    num_samples: int = 1,
    noise_std: float = 0.1,
) -> jax.Array:
    """Draw independent Gaussian time-domain noise samples."""
    shape = (num_samples, times.shape[0])
    noise = noise_std * jr.normal(key, shape)
    return noise[0] if num_samples == 1 else noise


def simulate_observations(
    key: jax.Array,
    times: jax.Array,
    params: Mapping[str, jax.Array],
    *,
    noise_std: float = 0.1,
) -> jax.Array:
    """Draw noise and add batched parametric signals."""
    n = next(iter(params.values())).shape[0]
    signals = get_signals(times, params)
    noise = draw_noise(key, times, num_samples=n, noise_std=noise_std)
    return noise + signals


def params_to_array(
    params: Mapping[str, jax.Array],
    names: Sequence[str] | None = None,
) -> tuple[jax.Array, list[str]]:
    names = list(params) if names is None else list(names)
    return jnp.stack([params[name] for name in names], axis=-1), names


def prior_standardization(
    priors: Mapping[str, Mapping[str, float | str]] = DEFAULT_PRIORS,
    names: Sequence[str] | None = None,
) -> tuple[jax.Array, jax.Array, list[str]]:
    """Analytic mean/std for independent uniform priors."""
    names = list(priors) if names is None else list(names)
    mean = []
    std = []
    for name in names:
        prior = priors[name]
        if prior["type"] != "uniform":
            raise ValueError(f"Unsupported prior type: {prior['type']}")
        prior_min = prior["min"]
        prior_max = prior["max"]
        mean.append(0.5 * (prior_min + prior_max))
        std.append((prior_max - prior_min) / jnp.sqrt(12.0))
    return jnp.asarray(mean), jnp.asarray(std), names


def standardize_parameters(
    params: Mapping[str, jax.Array],
    names: Sequence[str] | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, list[str]]:
    theta, names = params_to_array(params, names)
    mean = jnp.mean(theta, axis=0)
    std = jnp.std(theta, axis=0)
    return (theta - mean) / std, mean, std, names


def save_parameter_stats(
    path: str | Path,
    mean: jax.Array,
    std: jax.Array,
    names: Sequence[str],
) -> Path:
    """Save parameter mean/std arrays and a JSON sidecar with names."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, mean=np.asarray(mean), std=np.asarray(std), names=np.asarray(names))

    metadata_path = path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "names": list(names),
                "mean": np.asarray(mean).tolist(),
                "std": np.asarray(std).tolist(),
            },
            indent=2,
        )
    )
    return path
