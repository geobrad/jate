from typing import Any, Callable
import jax

Score = float
JaxTree = Any
PlayFunction = Callable[
    [JaxTree, JaxTree, int, jax.Array],
    tuple[float, float],
]
GPKernel = Callable[[jax.typing.ArrayLike, jax.typing.ArrayLike], jax.Array]
GPMeanFunction = Callable[[jax.typing.ArrayLike], jax.Array]
