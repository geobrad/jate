
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax


# Gauss–Hermite quadrature constants, computed once.
_x, _w0 = np.polynomial.hermite.hermgauss(40)
GH_X = jnp.asarray(_x)
GH_LOG_W0 = jnp.log(jnp.asarray(_w0))


def tilted_moments(m_c, v_c, w, n):
    """Moments of the tilted one-dimensional Gaussian distribution."""

    def newton_step(_, s0):
        p = jax.nn.sigmoid(s0)
        g = (w - n * p) - (s0 - m_c) / v_c
        curv = n * p * (1.0 - p) + 1.0 / v_c
        step = jnp.clip(
            g / curv,
            -3.0 * jnp.sqrt(v_c),
            3.0 * jnp.sqrt(v_c),
        )
        return s0 + step

    s0 = lax.fori_loop(0, 30, newton_step, m_c)

    p = jax.nn.sigmoid(s0)
    var = 1.0 / (n * p * (1.0 - p) + 1.0 / v_c)

    s = s0 + jnp.sqrt(2.0 * var) * GH_X
    h = (
        w * jax.nn.log_sigmoid(s)
        + (n - w) * jax.nn.log_sigmoid(-s)
        - (s - m_c) ** 2 / (2.0 * v_c)
    )

    logw = GH_LOG_W0 + h + GH_X**2
    weights = jax.nn.softmax(logw)

    m = jnp.sum(weights * s)
    v = jnp.sum(weights * (s - m) ** 2)
    return m, v


def ep_update(
    mu,
    Sigma,
    anchor_rating,
    games,
    c=1.0,
    max_sweeps=100,
    damping=1.0,
    tol=1e-6,
):
    """
    EP update of a Gaussian belief over player ratings.

    Parameters
    ----------
    mu : array, shape (n_players,)
        Prior mean.
    Sigma : array, shape (n_players, n_players)
        Prior covariance.
    games : tuple of arrays, each shape (m,)
        (i, j, w, n), where i and j are integer player indices,
        w is the score of i against j (draw = 0.5), and n is
        the number of games played between the pair.
    c : scalar
        Logit scale; P(i beats j) = sigmoid(c * (r_i - r_j)).
    max_sweeps : int
        Maximum number of EP sweeps.
    damping : scalar
        EP damping factor.
    tol : scalar
        Convergence tolerance.

    Returns
    -------
    mu_post, Sigma_post, n_sweeps
    """
    i_games, j_games, wins, counts = games
    m = i_games.shape[0]

    # One Gaussian site per game record.
    tau = jnp.zeros((m,), dtype=mu.dtype)
    nu = jnp.zeros((m,), dtype=mu.dtype)

    def sweep_cond(state):
        sweep, mu, Sigma, tau, nu, change = state
        # jax.debug.print("Sweep_cond {} {}", sweep, change)
        return (sweep < max_sweeps) & (change >= tol)

    def sweep_body(state):
        sweep, mu, Sigma, tau, nu, _ = state

        def game_body(k, carry):
            mu, Sigma, tau, nu, change = carry

            i = i_games[k]
            j = j_games[k]
            w = wins[k]
            n = counts[k]

            Sigma_i = jnp.where(i == -1, 0.0, Sigma[:, i])
            Sigma_j = jnp.where(j == -1, 0.0, Sigma[:, j])
            Su = c * (Sigma_i - Sigma_j)

            mu_i = jnp.where(i == -1, anchor_rating, mu[i])
            mu_j = jnp.where(j == -1, anchor_rating, mu[j])
            m_s = c * (mu_i - mu_j)

            Su_i = jnp.where(i == -1, 0.0, Su[i])
            Su_j = jnp.where(j == -1, 0.0, Su[j])
            v_s = c * (Su_i - Su_j)

            # prec_c = 1.0 / v_s - tau[k]

            denom_c = 1.0 - tau[k] * v_s

            def update(args):
                mu, Sigma, tau, nu, change = args


                safe_denom = jnp.maximum(denom_c, 1e-4)
                v_c = v_s / safe_denom
                m_c = (m_s - nu[k] * v_s) / safe_denom
                prec_c = safe_denom / v_s

                m_new, v_new = tilted_moments(m_c, v_c, w, n)

                tau_new = jnp.maximum(1.0 / v_new - prec_c, 0.0)
                nu_new = m_new / v_new - m_c * prec_c

                tau_u = (
                    (1.0 - damping) * tau[k]
                    + damping * tau_new
                )
                nu_u = (
                    (1.0 - damping) * nu[k]
                    + damping * nu_new
                )

                dt = tau_u - tau[k]
                dn = nu_u - nu[k]
                denom = 1.0 + dt * v_s

                mu = mu + Su * (dn - dt * m_s) / denom
                Sigma = (
                    Sigma - dt * jnp.outer(Su, Su) / denom
                )

                tau = tau.at[k].set(tau_u)
                nu = nu.at[k].set(nu_u)

                change = jnp.maximum(
                    change,
                    jnp.maximum(jnp.abs(dt), jnp.abs(dn)),
                )
                return mu, Sigma, tau, nu, change

            # Skip updates with invalid cavity precision.
            return lax.cond(
                (v_s > 1e-7) & (denom_c > 1e-4),
                update,
                lambda args: args,
                (mu, Sigma, tau, nu, change),
            )

        mu, Sigma, tau, nu, change = lax.fori_loop(
            0,
            m,
            game_body,
            (
                mu,
                Sigma,
                tau,
                nu,
                jnp.asarray(0.0, dtype=mu.dtype),
            ),
        )

        return sweep + 1, mu, Sigma, tau, nu, change

    state = (
        jnp.asarray(0, dtype=jnp.int32),
        mu,
        Sigma,
        tau,
        nu,
        jnp.asarray(jnp.inf, dtype=mu.dtype),
    )

    n_sweeps, mu, Sigma, tau, nu, _ = lax.while_loop(
        sweep_cond,
        sweep_body,
        state,
    )

    Sigma = (Sigma + Sigma.T) / 2.0
    return mu, Sigma, n_sweeps
