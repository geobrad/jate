from dataclasses import replace
import numpy as np
from scipy.special import expit as sigmoid
import numpy.typing as npt
from .state import JateState, is_full


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


def select_slot_to_evict(state: JateState) -> int:
    utility = snapshot_utilities(
        state.kappa,
        state.mu,
        state.anchor_rating,
    )
    return int(np.argmin(utility))


def evict(state: JateState, k: int) -> JateState:
    keep = np.arange(len(state.mu)) != k
    # print("keep", keep.shape)
    # print("mu", state.mu.shape)
    # print("Sigma", state.Sigma.shape)

    return replace(
        state,
        snapshots=state.snapshots[:k] + state.snapshots[k + 1:],
        times=state.times[keep],
        mu=state.mu[keep],
        Sigma=state.Sigma[keep][:, keep],
    )


def evict_one(state: JateState) -> JateState:
    slot_to_evict = select_slot_to_evict(state)
    return evict(state, slot_to_evict)


def push(state: JateState, t: npt.ArrayLike, new_snapshot: object) -> JateState:
    if is_full(state):
        state = evict_one(state)

    last = len(state.mu) - 1

    mu_new = state.mu[last] if last >= 0 else state.anchor_rating
    mu_next = np.concatenate([state.mu, [mu_new]])

    cov_new = state.Sigma[:, last] if last >= 0 else np.zeros(len(state.mu))
    var_last = state.Sigma[last, last] if last >= 0 else 0.0
    var_new = var_last + state.snapshot_variance

    Sigma_next = np.concatenate([
        np.concatenate([state.Sigma, cov_new[:, None]], axis=1),
        np.concatenate([cov_new[None, :], [[var_new]]], axis=1)
    ])

    return replace(
        state,
        snapshots=state.snapshots + [new_snapshot],
        times=np.concatenate([state.times, np.asarray([t])]),
        mu=mu_next,
        Sigma=Sigma_next,
    )
