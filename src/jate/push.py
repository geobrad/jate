import numpy as np
from scipy.special import expit as sigmoid
import numpy.typing as npt


def snapshot_utilities(
    kappa: float,
    mu: npt.NDArray[np.float64],
    anchor_rating: float,
) -> npt.NDArray[np.float64]:
    target = len(mu) - 1

    """Estimate snapshot utility from its contribution to target-anchor conductance."""
    n = mu.shape[0]
    idx = np.arange(n)

    # Expected internal Fisher information between buffer slots
    delta = mu[:, None] - mu[None, :]
    p = sigmoid(kappa * delta)
    g = kappa**2 * p * (1.0 - p)
    np.fill_diagonal(g, 0.0)

    # Expected fisher information from each slot to the external anchor
    p_anchor = sigmoid(kappa * (mu - anchor_rating))
    g_anchor = kappa**2 * p_anchor * (1.0 - p_anchor)

    # Weighted graph Laplacian.
    L = -g
    np.fill_diagonal(L, np.sum(g, axis=1) + g_anchor)

    # Anchor at voltage 0; target at voltage 1.
    free = idx != target

    A = np.where(free[:, None] & free[None, :], L, 0.0)
    np.fill_diagonal(A, np.where(free, np.diagonal(L), 1.0))

    # Boundary contribution from the target's fixed voltage.
    b = np.where(free, g[:, target], (idx == target).astype(mu.dtype))

    voltage = np.linalg.solve(A, b)

    # Sum incident-edge energy for each vertex.
    dv = voltage[:, None] - voltage[None, :]
    internal_energy = np.sum(g * dv**2, axis=1)
    anchor_energy = g_anchor * voltage**2
    utility = internal_energy + anchor_energy

    return np.where(free, utility, np.inf)
