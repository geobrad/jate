import jax
import jax.numpy as jnp
from .types import JaxTree, PlayFunction
from .state import JateState
from . import kernels
from .mean_functions import constant


def init(
        max_snapshots: int,
        kappa: jax.typing.ArrayLike,
        snapshot_zero: JaxTree,
        snapshot_zero_rating: float,
        variance: jax.typing.ArrayLike,
        lengthscale: jax.typing.ArrayLike,
        snapshot_variance: jax.typing.ArrayLike,
        play_fn: PlayFunction,
        rng_key: jax.Array,
) -> JateState:
    """
    Initializes the JateState with default values.
    Returns:
        JateState: An instance of JateState with default values.
    """
    kernel = kernels.integrated_matern32(variance, lengthscale)
    # kernel = kernels.matern32(variance, lengthscale)
    # kernel = kernels.white_noise(variance)
    return JateState(
        anchor_snapshot=snapshot_zero,
        anchor_rating=jnp.asarray(snapshot_zero_rating, dtype=float),
        n_slots=max_snapshots,
        kappa=kappa,
        snapshots=jax.tree.map(
            lambda x: jnp.broadcast_to(
                x, (max_snapshots, *jnp.asarray(x).shape)),
            snapshot_zero,
        ),
        mask=jnp.full((max_snapshots,), False),
        times=jnp.full((max_snapshots,), 0),
        mu=jnp.zeros(max_snapshots),
        Sigma=jnp.eye(max_snapshots),
        next_slot=jnp.array(0),
        last_slot=jnp.array(-1),
        kernel=kernel,
        mean_fn=constant(snapshot_zero_rating),
        snapshot_variance=snapshot_variance,
        play_fn=play_fn,
        rng_key=rng_key,
        n_played=jnp.array(0),
    )
