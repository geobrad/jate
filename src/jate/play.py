from dataclasses import replace
import jax
import jax.numpy as jnp
from .state import JateState
from .ep_update import ep_update


def score_pairing(
    state: JateState,
    n_games: int,
    m: jax.typing.ArrayLike,
    i: jax.typing.ArrayLike,
    j: jax.typing.ArrayLike,
) -> jax.Array:
    # Expected game outcome under the current Gaussian mean.
    r_i = jnp.where(i == -1, state.anchor_rating, state.mu[i])
    r_j = jnp.where(j == -1, state.anchor_rating, state.mu[j])

    cov_mi = jnp.where(i == -1, 0, state.Sigma[m, i])
    cov_mj = jnp.where(j == -1, 0.0, state.Sigma[m, j])
    cov_ij = jnp.where(i == -1, 0.0, jnp.where(j == -1, 0.0, state.Sigma[i, j]))
    var_i = jnp.where(i == -1, 0, state.Sigma[i, i])
    var_j = jnp.where(j == -1, 0, state.Sigma[j, j])
    valid_i = jnp.where(i == -1, True, state.mask[i])
    valid_j = jnp.where(j == -1, True, state.mask[j])

    p = jax.nn.sigmoid(state.kappa * (r_i - r_j))
    w = 2 * n_games * p * (1.0 - p)

    # Covariance between the target rating and the rating difference.
    c = state.kappa * (cov_mi - cov_mj)

    # Variance of the rating difference.
    v = var_i + var_j - 2.0 * cov_ij

    s = w * c**2 / (1.0 + w * state.kappa**2 * v)

    return jnp.where(valid_i & valid_j, s, 0.0)


def select_pairing(state: JateState, n_games: int = 1) -> tuple[jax.Array, jax.Array]:
    i_slots, j_slots = jnp.triu_indices(state.n_slots, k=1)
    i = jnp.concatenate([jnp.full(state.n_slots, -1), i_slots])
    j = jnp.concatenate([jnp.arange(state.n_slots), j_slots])
    pairing_scores = score_pairing(state, n_games, state.last_slot, i, j)
    best = jnp.argmax(pairing_scores)
    return i[best], j[best]


def play_one_pair(state: JateState, n_games: int) -> JateState:
    i, j = select_pairing(state)
    # jax.debug.print("Playing {} vs {}", state.times[i], state.times[j])
    snapshot_i = jax.lax.cond(
        i == -1,
        lambda: state.anchor_snapshot,
        lambda: jax.tree.map(lambda x: x[i], state.snapshots),
    )
    snapshot_j = jax.tree.map(lambda x: x[j], state.snapshots)
    ij_key, ji_key, out_key = jax.random.split(state.rng_key, 3)
    i_score_ij, _ = state.play_fn(snapshot_i, snapshot_j, n_games, ij_key)
    _, i_score_ji = state.play_fn(snapshot_j, snapshot_i, n_games, ji_key)

    games = (
        jnp.array([i]),
        jnp.array([j]),
        jnp.array([i_score_ij + i_score_ji]),
        jnp.array([n_games * 2]),
    )
    mu_new, Sigma_new, n_sweeps = ep_update(
        state.mu,
        state.Sigma,
        state.anchor_rating,
        games,
        c=state.kappa,
    )
    # jax.debug.print("Sweeps: {}", n_sweeps)

    return replace(
        state,
        mu=mu_new,
        Sigma=Sigma_new,
        rng_key=out_key,
        n_played=state.n_played + n_games * 2,
    )


def play(state: JateState, n_pairs: int, n_games: int) -> JateState:
    jax.debug.print("Playing {} pairs of {} games", n_pairs, n_games)
    return jax.lax.cond(
        state.last_slot < 0,
        lambda s: s,
        lambda s: jax.lax.fori_loop(
            0,
            n_pairs,
            lambda _, state_i: play_one_pair(state_i, n_games),
            s,
        ),
        state,
    )


def play_until(state: JateState, target_variance: jax.typing.ArrayLike, n_games: int) -> JateState:
    return jax.lax.cond(
        state.last_slot < 0,
        lambda s: s,
        lambda s: jax.lax.while_loop(
            lambda state_i: state_i.Sigma[state_i.last_slot, state_i.last_slot] > target_variance,
            lambda state_i: play_one_pair(state_i, n_games),
            s,
        ),
        state,
    )
