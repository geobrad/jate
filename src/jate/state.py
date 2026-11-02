from dataclasses import dataclass
import jax
import jax.numpy as jnp
from .types import JaxTree, PlayFunction, GPKernel, GPMeanFunction


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class JateState:
    """A named tuple representing the state of the jate system."""
    anchor_snapshot: JaxTree
    anchor_rating: jax.Array  # Scalar float
    n_slots: int = jax.tree.static()
    snapshots: JaxTree
    mask: jax.Array  # Shape (n,) of booleans
    times: jax.Array  # Shape (n,) of int32
    mu: jax.Array  # Shape (n,) of float
    Sigma: jax.Array  # Shape (n, n) of float
    last_slot: jax.Array  # Scalar int; -1 if empty
    next_slot: jax.Array  # Scalar int; -1 if full
    snapshot_variance: jax.Array
    play_fn: PlayFunction = jax.tree.static()
    n_played: jax.Array
    rng_key: jax.Array
    kappa: jax.typing.ArrayLike = jax.tree.static()


def is_full(state: JateState) -> bool:
    return jnp.sum(state.mask) == state.n_slots


def belief(state: JateState):
    return (
        state.mask,
        state.times,
        state.mu,
        jnp.diagonal(state.Sigma),
    )


def games_played(state: JateState):
    return state.n_played
