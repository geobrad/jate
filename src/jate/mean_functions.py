import jax
import jax.numpy as jnp
from .types import GPMeanFunction

def constant(value: jax.typing.ArrayLike) -> GPMeanFunction:
    return lambda x: jnp.broadcast_to(value, x.shape)
