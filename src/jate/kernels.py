import jax
import jax.numpy as jnp
from .types import GPKernel


def matern32(
    lengthscale: jax.typing.ArrayLike,
    variance: jax.typing.ArrayLike,
) -> GPKernel:
    def kernel(x1: jax.Array, x2: jax.Array) -> jax.Array:
        r = jnp.abs(x1 - x2) / lengthscale
        z = jnp.sqrt(3.0) * r
        return variance * (1.0 + z) * jnp.exp(-z)
    return kernel


def integrated_matern32(
    variance: jax.typing.ArrayLike,
    lengthscale: jax.typing.ArrayLike,
) -> GPKernel:
    a = jnp.sqrt(3.0) / lengthscale

    def kernel(t1: jax.Array, t2: jax.Array) -> jax.Array:
        m = jnp.minimum(t1, t2)
        M = jnp.maximum(t1, t2)

        am = a * m
        aM = a * M
        ad = a * (M - m)

        def J(x):
            return 2.0 * x - 3.0 + (x + 3.0) * jnp.exp(-x)

        return variance / a**2 * (
            J(am) + J(aM) - J(ad)
        )

    return kernel


def white_noise(variance: jax.typing.ArrayLike) -> GPKernel:
    return lambda x1, x2: jnp.where(x1 == x2, 0.0, variance)
