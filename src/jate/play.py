import numpy as np
from .ep_update import ep_update
import numpy.typing as npt
from scipy.special import expit as sigmoid


def score_pairing(
    anchor_rating: float,
    mu: npt.NDArray,
    Sigma: npt.NDArray,
    kappa: float,
    n_games: int,
    m: npt.ArrayLike,
    i: npt.ArrayLike,
    j: npt.ArrayLike,
) -> npt.NDArray:

    i = np.asarray(i, dtype=np.intp)
    j = np.asarray(j, dtype=np.intp)
    m = np.asarray(m, dtype=np.intp)

    # Expected game outcome under the current Gaussian mean.
    r_i = np.where(i == -1, anchor_rating, mu[i])
    r_j = np.where(j == -1, anchor_rating, mu[j])

    cov_mi = np.where(i == -1, 0, Sigma[m, i])
    cov_mj = np.where(j == -1, 0.0, Sigma[m, j])
    cov_ij = np.where(i == -1, 0.0, np.where(j == -1, 0.0, Sigma[i, j]))
    var_i = np.where(i == -1, 0, Sigma[i, i])
    var_j = np.where(j == -1, 0, Sigma[j, j])

    p = sigmoid(kappa * (r_i - r_j))
    w = 2 * n_games * p * (1.0 - p)

    # Covariance between the target rating and the rating difference.
    c = kappa * (cov_mi - cov_mj)

    # Variance of the rating difference.
    v = var_i + var_j - 2.0 * cov_ij

    s = w * c**2 / (1.0 + w * kappa**2 * v)

    return s


def best_pairing(
    anchor_rating: float,
    mu: npt.NDArray,
    Sigma: npt.NDArray,
    kappa: float,
) -> tuple[npt.NDArray, npt.NDArray]:
    n = len(mu)
    i_slots, j_slots = np.triu_indices(n, k=1)
    i = np.concatenate([np.full(n, -1), i_slots])
    j = np.concatenate([np.arange(n), j_slots])
    pairing_scores = score_pairing(
        anchor_rating,
        mu,
        Sigma,
        kappa,
        1,
        n - 1,
        i,
        j)
    best = np.argmax(pairing_scores)
    return i[best], j[best]
