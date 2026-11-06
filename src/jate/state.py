from dataclasses import dataclass
from typing import Callable
import numpy as np
from .types import PlayFunction
import numpy.typing as npt


@dataclass(frozen=True)
class JateState[Snapshot, RNGKey]:
    """A named tuple representing the state of the jate system."""
    anchor_snapshot: Snapshot
    anchor_rating: float
    max_snapshots: int
    snapshots: list[Snapshot]
    times: npt.NDArray[np.float64]  # Shape (n,)
    mu: npt.NDArray[np.float64]  # Shape (n,)
    Sigma: npt.NDArray[np.float64]  # Shape (n, n)
    snapshot_variance: float
    play_fn: PlayFunction[Snapshot, RNGKey]
    n_played: int
    kappa: float
    rng_key_split_fn: Callable[[RNGKey], tuple[RNGKey, RNGKey]]
    rng_key: RNGKey


def is_full(state: JateState) -> bool:
    return len(state.snapshots) == state.max_snapshots


def belief(state: JateState):
    return (
        state.times,
        state.mu,
        np.diagonal(state.Sigma),
    )


def games_played(state: JateState):
    return state.n_played
