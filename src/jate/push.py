from dataclasses import replace
import jax
import jax.numpy as jnp
from .types import JaxTree
from .state import JateState


KAPPA = jnp.log(10.0) / 400.0


def snapshot_utilities(
    kappa: jax.typing.ArrayLike,
    mu: jax.Array,
    target: jax.Array,
    anchor_rating: jax.Array,
) -> jax.Array:
    """Estimate snapshot utility from its contribution to target-anchor conductance."""
    n = mu.shape[0]
    idx = jnp.arange(n)

    # Expected internal Fisher information between buffer slots
    delta = mu[:, None] - mu[None, :]
    p = jax.nn.sigmoid(kappa * delta)
    g = kappa**2 * p * (1.0 - p)
    g = g.at[idx, idx].set(0.0)

    # Expected fisher information from each slot to the external anchor
    p_anchor = jax.nn.sigmoid(kappa * (mu - anchor_rating))
    g_anchor = kappa**2 * p_anchor * (1.0 - p_anchor)

    # Weighted graph Laplacian.
    L = -g
    L = L.at[idx, idx].set(jnp.sum(g, axis=1) + g_anchor)

    # Anchor at voltage 0; target at voltage 1.
    free = idx != target

    A = jnp.where(free[:, None] & free[None, :], L, 0.0)
    A = A.at[idx, idx].set(jnp.where(free, jnp.diag(L), 1.0))

    # Boundary contribution from the target's fixed voltage.
    b = jnp.where(free, g[:, target], (idx == target).astype(mu.dtype))

    voltage = jnp.linalg.solve(A, b)

    # Sum incident-edge energy for each vertex.
    dv = voltage[:, None] - voltage[None, :]
    internal_energy = jnp.sum(g * dv**2, axis=1)
    anchor_energy = g_anchor * voltage**2
    utility = internal_energy + anchor_energy

    return jnp.where(free, utility, jnp.inf)


def mask1d(mask: jax.typing.ArrayLike, x: jax.typing.ArrayLike):
    return jnp.where(mask, x, 0)


def mask2d(mask: jax.typing.ArrayLike, X: jax.typing.ArrayLike):
    return jnp.where(mask[:, None] & mask[None, :], X, 0)


def is_full(state: JateState) -> bool:
    return jnp.sum(state.mask) == state.n_slots


def select_slot_to_evict(state: JateState) -> int:
    utility = snapshot_utilities(
        state.kappa,
        state.mu,
        state.last_slot,
        state.anchor_rating,
    )
    return jnp.argmin(utility)


def evict(state: JateState, k: int) -> JateState:
    return replace(
        state,
        mask=state.mask.at[k].set(False),
        next_slot=k,
    )


def evict_one(state: JateState) -> JateState:
    slot_to_evict = select_slot_to_evict(state)
    return evict(state, slot_to_evict)


@jax.jit
def push(state: JateState, t: jax.typing.ArrayLike, new_snapshot: JaxTree) -> JateState:
    state = jax.lax.cond(
        state.next_slot == -1,
        evict_one,
        lambda s: s,
        state,
    )

    # T = state.times

    # K_unmasked = state.kernel(T[:, None], T[None, :])
    # K = mask2d(state.mask, K_unmasked) + jnp.diag(~state.mask)
    # k_t = mask1d(state.mask, state.kernel(T, t))
    # k_tt = state.kernel(t, t)
    # mu_T_prior = state.mean_fn(T)
    # mu_t_prior = state.mean_fn(t)

    # weights = jnp.linalg.solve(K, k_t)

    # mu_t = mu_t_prior + weights @ mask1d(state.mask, state.mu - mu_T_prior)

    # Sigma = mask2d(state.mask, state.Sigma)

    # cov_t = jnp.where(state.last_slot < 0,
    #                   jnp.zeros_like(k_t), Sigma @ weights)
    # var_t = jnp.where(state.last_slot < 0, k_tt, k_tt - k_t @
    #                   weights + weights @ Sigma @ weights)

    j = state.next_slot
    # mu_next = state.mu.at[j].set(mu_t)
    # Sigma_next = Sigma \
    #     .at[j, :].set(cov_t) \
    #     .at[:, j].set(cov_t) \
    #     .at[j, j].set(var_t + state.snapshot_variance)

    mu_next = state.mu.at[j].set(jnp.where(state.last_slot < 0, state.anchor_rating, state.mu[state.last_slot]))
    Sigma_next = state.Sigma.at[j, :].set(jnp.zeros(state.n_slots)).at[:, j].set(jnp.zeros(state.n_slots)).at[j, j].set(state.snapshot_variance)


    return replace(
        state,
        snapshots=jax.tree.map(
            lambda x, y: x.at[j].set(y),
            state.snapshots,
            new_snapshot,
        ),
        mask=state.mask.at[j].set(True),
        times=state.times.at[j].set(t),
        mu=mu_next,
        Sigma=Sigma_next,
        last_slot=j,
        next_slot=jnp.where(
            (j + 1 < state.n_slots) & ~state.mask[j + 1],
            j + 1,
            -1,
        ),
    )
