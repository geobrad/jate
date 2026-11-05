from typing import Any, Callable
from numpy.typing import NDArray, ArrayLike
import numpy.typing as npt
from optree.typing import PyTree

type Score = float
type Snapshot = Any
type PlayFunction[RNGKey] = Callable[
    [Snapshot, Snapshot, int, RNGKey],
    tuple[float, float],
]
type GPKernel = Callable[[ArrayLike, ArrayLike], NDArray]
type GPMeanFunction = Callable[[ArrayLike], NDArray]
