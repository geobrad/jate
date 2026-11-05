"""Expectation Propagation update of a Gaussian belief over ratings, given new game results.

    ep_update(mu, Sigma, anchor_rating, games) -> (mu_post, Sigma_post, n_sweeps)

State carried between calls is only (mu, Sigma). EP "sites" are scratch: they start at zero on
every call and are discarded afterwards.
"""
import numpy as np
from scipy.special import log_expit

x, w0 = np.polynomial.hermite.hermgauss(40)


def tilted_moments(m_c, v_c, W, N):
    """Mean and variance of  f(s) * Normal(s; m_c, v_c),  f(s) = sigmoid(s)^W * sigmoid(-s)^(N-W).

    Gauss-Hermite quadrature centred on the *tilted* distribution (mode found by a few 1-D Newton
    steps), not on the cavity. Centring on the cavity fails when the cavity is diffuse relative to
    the likelihood (e.g. a brand-new snapshot with a wide prior).
    """
    s0 = m_c
    for _ in range(30):
        p = 1.0 / (1.0 + np.exp(-s0))
        g = (W - N * p) - (s0 - m_c) / v_c
        curv = N * p * (1 - p) + 1.0 / v_c
        s0 = s0 + np.clip(g / curv, -3 * np.sqrt(v_c), 3 * np.sqrt(v_c))
    p = 1.0 / (1.0 + np.exp(-s0))
    var = 1.0 / (N * p * (1 - p) + 1.0 / v_c)
    s = s0 + np.sqrt(2 * var) * x
    h = W * log_expit(s) + (N - W) * log_expit(-s) - (s - m_c) ** 2 / (2 * v_c)
    logw = np.log(w0) + h + x ** 2          # x**2 undoes the Gauss-Hermite weight exp(-x**2)
    w = np.exp(logw - logw.max())
    w /= w.sum()
    m = w @ s
    return m, w @ (s - m) ** 2


def ep_update(mu, Sigma, anchor_rating, games, c=1.0, max_sweeps=100, damping=1.0, tol=1e-8):
    """
    mu, Sigma : prior Gaussian belief over ratings (n,), (n, n)
    anchor_rating : deterministic rating of the player with index -1
    games     : iterable of (i, j, W, N) -- one entry per unordered pair that played in this batch;
                i and/or j may be -1; W = score of i against j, N = number of games
    c         : logit scale; P(i beats j) = sigmoid(c * (r_i - r_j)).  c = ln(10)/400 for Elo units.
    """
    mu, Sigma = np.array(mu, dtype=float), np.array(Sigma, dtype=float)
    anchor_rating = np.asarray(anchor_rating, dtype=float).item()
    games = list(games)
    tau, nu = np.zeros(len(games)), np.zeros(len(games))     # site precisions / precision-means
    n_sweeps = 0
    for n_sweeps in range(1, max_sweeps + 1):
        change = 0.0
        for k, (i, j, W, N) in enumerate(games):
            if i < -1 or j < -1:
                raise IndexError("Player indices must be -1 or valid indices into mu.")
            Su = np.zeros_like(mu)
            m_s = 0.0
            if i == -1:
                m_s += c * anchor_rating
            else:
                Su += c * Sigma[:, i]
                m_s += c * mu[i]
            if j == -1:
                m_s -= c * anchor_rating
            else:
                Su -= c * Sigma[:, j]
                m_s -= c * mu[j]
            v_s = c * ((Su[i] if i >= 0 else 0.0) - (Su[j] if j >= 0 else 0.0))
            if v_s <= 0.0:
                continue
            prec_c = 1.0 / v_s - tau[k]                       # cavity precision
            if prec_c <= 1e-12:
                continue
            v_c = 1.0 / prec_c
            m_c = v_c * (m_s / v_s - nu[k])                   # cavity mean
            m_new, v_new = tilted_moments(m_c, v_c, W, N)
            tau_new = max(1.0 / v_new - prec_c, 0.0)
            nu_new = m_new / v_new - m_c * prec_c
            tau_u = (1 - damping) * tau[k] + damping * tau_new
            nu_u = (1 - damping) * nu[k] + damping * nu_new
            dt, dn = tau_u - tau[k], nu_u - nu[k]
            denom = 1.0 + dt * v_s
            mu = mu + Su * (dn - dt * m_s) / denom            # rank-one mean update
            Sigma = Sigma - dt * np.outer(Su, Su) / denom     # Sherman-Morrison
            tau[k], nu[k] = tau_u, nu_u
            change = max(change, abs(dt), abs(dn))
        if change < tol:
            break
    return mu, (Sigma + Sigma.T) / 2, n_sweeps
