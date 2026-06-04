import jax
import jax.numpy as jnp

from types import SimpleNamespace

from flowjax.flows import coupling_flow
from flowjax.distributions import Normal
from flowjax.bijections import Affine

from dataclasses import dataclass, field

@dataclass
class Config:
    param_dim: int = 2
    conditioning_dim: int = 256
    flow_layers: int = 6
    nn_width: int = 64
    nn_depth: int = 2

    @property
    def freqs(self):
        return jnp.linspace(1e-4, 1e-1, self.conditioning_dim)

    @property
    def cond_dim(self):
        return self.conditioning_dim * 2
