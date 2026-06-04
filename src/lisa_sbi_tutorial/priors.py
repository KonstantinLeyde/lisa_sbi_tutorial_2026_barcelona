import jax
import jax.numpy as jnp
import jax.random as jr
from typing import Mapping


class PriorModel:
    """Draws samples from a set of independent priors.

    Usage
    -----
    prior = PriorModel(prior_bounds)
    samples = prior.sample(jr.key(0), num_samples=10_000)
    # {'snr': Array(...), 'frequency': Array(...), ...}
    """

    SUPPORTED = {"uniform", "log_uniform", "normal"}

    def __init__(self, priors: Mapping[str, Mapping[str, float | str]]):
        unsupported = {p["type"] for p in priors.values()} - self.SUPPORTED
        if unsupported:
            raise ValueError(f"Unsupported prior type(s): {unsupported}")
        self.priors = dict(priors)
        self.param_names = list(priors.keys())
        self._n_params = len(self.param_names)  # concrete int, safe under JIT

    # ------------------------------------------------------------------ #
    # Sampling                                                             #
    # ------------------------------------------------------------------ #

    def sample(
        self,
        key: jax.Array,
        batch_shape: tuple[int, ...] = (1000,),
        return_array: bool = False,
    ) -> dict[str, jax.Array]:
        """Draw independent samples from each prior.

        Parameters
        ----------
        key:         JAX PRNG key
        batch_shape: shape of the output batch, e.g. (1000,) or (32, 16)

        Returns
        -------
        dict mapping param name -> Array of shape (*batch_shape)
        """
        keys = jr.split(key, self._n_params)
        samples = {
            name: self._draw_one(subkey, prior, batch_shape)
            for subkey, (name, prior) in zip(keys, self.priors.items())
        }
        if return_array:
            return jnp.stack([samples[name] for name in self.param_names], axis=-1)
        return samples

    def _draw_one(self, key: jax.Array, prior: dict, batch_shape: tuple) -> jax.Array:
        t = prior["type"]
        if t == "uniform":
            return jr.uniform(key, shape=batch_shape, minval=prior["min"], maxval=prior["max"])
        elif t == "log_uniform":
            log_min, log_max = jnp.log(prior["min"]), jnp.log(prior["max"])
            return jnp.exp(jr.uniform(key, shape=batch_shape, minval=log_min, maxval=log_max))
        elif t == "normal":
            return jr.normal(key, shape=batch_shape) * prior["std"] + prior["mean"]
