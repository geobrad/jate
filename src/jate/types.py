from typing import Callable

type Score = float
type PlayFunction[Snapshot, RNGKey] = Callable[
    [Snapshot, Snapshot, int, RNGKey],
    tuple[float, float],
]
