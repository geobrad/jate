import numpy as np
from .types import PlayFunction
from .state import JateState
from typing import Callable


def init[Snapshot, RNGKey](
        max_snapshots: int,
        kappa: float,
        snapshot_zero: Snapshot,
        snapshot_zero_rating: float,
        snapshot_variance: float,
        play_fn: PlayFunction[Snapshot, RNGKey],
        rng_key_split_fn: Callable[[RNGKey], tuple[RNGKey, RNGKey]],
        rng_key: RNGKey,
) -> JateState:
    """
    Initializes the JateState with default values.
    Returns:
        JateState: An instance of JateState with default values.
    """
    return JateState[Snapshot, RNGKey](
        anchor_snapshot=snapshot_zero,
        anchor_rating=np.asarray(snapshot_zero_rating, dtype=float),
        max_snapshots=max_snapshots,
        kappa=kappa,
        snapshots=[],
        times=np.zeros((0,)),
        mu=np.zeros((0,)),
        Sigma=np.eye(0),
        snapshot_variance=snapshot_variance,
        play_fn=play_fn,
        n_played=0,
        rng_key_split_fn=rng_key_split_fn,
        rng_key=rng_key,
    )
