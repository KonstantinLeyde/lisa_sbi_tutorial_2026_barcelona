import jax
import jax.numpy as jnp
import jax.random as jr


class NoiseDataset:

    def __init__(self, freqencies):

        self.frequencies = freqencies

    def sample(
        self,
        key: jax.Array,
        batch_shape: tuple,
    ) -> jax.Array:
        
        sample_shape = batch_shape + self.frequencies.shape
        noise = jr.normal(key, shape=sample_shape)
        
        return noise
