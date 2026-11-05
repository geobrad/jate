from dataclasses import replace
import numpy as np
from .state import JateState
from .ep_update import ep_update
import numpy.typing as npt
from scipy.special import expit as sigmoid

def score_pairing(
    state: JateState,
    n_games: int,
    m: npt.ArrayLike,
    i: npt.ArrayLike,
    j: npt.ArrayLike,
) -> npt.NDArray:

    i = np.asarray(i, dtype=np.intp)
    j = np.asarray(j, dtype=np.intp)
    m = np.asarray(m, dtype=np.intp)

    # Expected game outcome under the current Gaussian mean.
    r_i = np.where(i == -1, state.anchor_rating, state.mu[i])
    r_j = np.where(j == -1, state.anchor_rating, state.mu[j])

    cov_mi = np.where(i == -1, 0, state.Sigma[m, i])
    cov_mj = np.where(j == -1, 0.0, state.Sigma[m, j])
    cov_ij = np.where(i == -1, 0.0, np.where(j == -1, 0.0, state.Sigma[i, j]))
    var_i = np.where(i == -1, 0, state.Sigma[i, i])
    var_j = np.where(j == -1, 0, state.Sigma[j, j])

    p = sigmoid(state.kappa * (r_i - r_j))
    w = 2 * n_games * p * (1.0 - p)

    # Covariance between the target rating and the rating difference.
    c = state.kappa * (cov_mi - cov_mj)

    # Variance of the rating difference.
    v = var_i + var_j - 2.0 * cov_ij

    s = w * c**2 / (1.0 + w * state.kappa**2 * v)

    return s


def select_pairing(state: JateState, n_games: int = 1) -> tuple[npt.NDArray, npt.NDArray]:
    n = len(state.mu)
    i_slots, j_slots = np.triu_indices(n, k=1)
    i = np.concatenate([np.full(n, -1), i_slots])
    j = np.concatenate([np.arange(n), j_slots])
    pairing_scores = score_pairing(state, n_games, n - 1, i, j)
    best = np.argmax(pairing_scores)
    return i[best], j[best]


def play_one_pair(state: JateState, n_games: int) -> JateState:
    # print("Selecting pairing...")
    i, j = select_pairing(state)
    # jax.debug.print("Playing {} vs {}", state.times[i], state.times[j])
    # print("Playing...")
    snapshot_i = state.snapshots[i] if i >= 0 else state.anchor_snapshot
    snapshot_j = state.snapshots[j]
    key_next, key_ij = state.rng_key_split_fn(state.rng_key)
    key_next, key_ji = state.rng_key_split_fn(key_next)
    i_score_ij, _ = state.play_fn(snapshot_i, snapshot_j, n_games, key_ij)
    _, i_score_ji = state.play_fn(snapshot_j, snapshot_i, n_games, key_ji)

    # games = (
    #     np.array([i]),
    #     np.array([j]),
    #     np.array([i_score_ij + i_score_ji]),
    #     np.array([n_games * 2]),
    # )
    # print("Updating belief...")
    games = [(i, j, i_score_ij + i_score_ji, n_games * 2)]
    mu_new, Sigma_new, n_sweeps = ep_update(
        state.mu,
        state.Sigma,
        state.anchor_rating,
        games,
        c=state.kappa,
    )
    if n_sweeps >= 100:
        print("Warning: EP did not converge after 100 sweeps.")
    # jax.debug.print("Sweeps: {}", n_sweeps)

    return replace(
        state,
        mu=mu_new,
        Sigma=Sigma_new,
        n_played=state.n_played + n_games * 2,
        rng_key=key_next,
    )


def play(state: JateState, n_pairs: int, n_games: int) -> JateState:
    if len(state.mu) == 0:
        return state
    # jax.debug.print("Playing {} pairs of {} games", n_pairs, n_games)
    for i in range(n_pairs):
        state = play_one_pair(state, n_games)
    return state


def play_until(state: JateState, target_variance: npt.ArrayLike, n_games: int) -> JateState:
    if len(state.mu) == 0:
        return state
    while state.Sigma[-1, -1] > target_variance:
        state = play_one_pair(state, n_games)
    return state
