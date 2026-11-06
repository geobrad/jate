from typing import Callable
import numpy as np
import numpy.typing as npt
from .types import PlayFunction
from .ep_update import ep_update
from .push import snapshot_utilities
from .play import best_pairing


class Jate[Snapshot, RNGKey]:
    n_snapshots: int = 0
    anchor_snapshot: Snapshot
    anchor_rating: float
    max_snapshots: int
    snapshots: list[Snapshot] = []
    times = np.zeros(0)  # Shape (n,)
    mu = np.zeros(0)  # Shape (n,)
    Sigma = np.eye(0)  # Shape (n, n)
    snapshot_variance: float
    play_fn: PlayFunction[Snapshot, RNGKey]
    n_played: int = 0
    kappa: float
    rng_key_split_fn: Callable[[RNGKey], tuple[RNGKey, RNGKey]]
    rng_key: RNGKey

    def __init__(
        self,
        max_snapshots: int,
        kappa: float,
        snapshot_zero: Snapshot,
        snapshot_zero_rating: float,
        snapshot_variance: float,
        play_fn: PlayFunction[Snapshot, RNGKey],
        rng_key_split_fn: Callable[[RNGKey], tuple[RNGKey, RNGKey]],
        rng_key: RNGKey,
    ):
        self.anchor_snapshot = snapshot_zero
        self.anchor_rating = snapshot_zero_rating
        self.max_snapshots = max_snapshots
        self.kappa = kappa
        self.snapshot_variance = snapshot_variance
        self.play_fn = play_fn
        self.rng_key_split_fn = rng_key_split_fn
        self.rng_key = rng_key

    def is_full(self) -> bool:
        return self.n_snapshots == self.max_snapshots

    def is_empty(self) -> bool:
        return self.n_snapshots == 0

    def belief(self):
        return (
            self.times,
            self.mu,
            np.diagonal(self.Sigma),
        )

    def games_played(self):
        return self.n_played

    def random_key(self) -> RNGKey:
        self.rng_key, key = self.rng_key_split_fn(self.rng_key)
        return key

    def evict_one(self) -> tuple[float, Snapshot]:
        if self.is_empty():
            raise ValueError("Cannot evict snapshot: Jate is empty.")
        slot_utilities = snapshot_utilities(
            self.kappa,
            self.mu,
            self.anchor_rating,
        )
        slot_to_evict = int(np.argmin(slot_utilities))
        return self.evict(slot_to_evict)

    def evict(self, k: int) -> tuple[float, Snapshot]:
        evicted_time = self.times[k]
        evicted_snapshot = self.snapshots[k]
        keep = np.arange(self.n_snapshots) != k
        self.n_snapshots -= 1
        self.snapshots = self.snapshots[:k] + self.snapshots[k + 1:]
        self.times = self.times[keep]
        self.mu = self.mu[keep]
        self.Sigma = self.Sigma[keep][:, keep]
        return evicted_time, evicted_snapshot

    def push(self, snapshot: Snapshot, time: float) -> tuple[float, Snapshot] | None:
        evicted = self.evict_one() if self.is_full() else None
        last = self.n_snapshots - 1

        mu_new = self.mu[last] if last >= 0 else self.anchor_rating
        mu_next = np.concatenate([self.mu, [mu_new]])

        cov_new = self.Sigma[:, last] if last >= 0 else np.zeros(
            self.n_snapshots)
        var_last = self.Sigma[last, last] if last >= 0 else 0.0
        var_new = var_last + self.snapshot_variance

        Sigma_next = np.concatenate([
            np.concatenate([self.Sigma, cov_new[:, None]], axis=1),
            np.concatenate([cov_new[None, :], [[var_new]]], axis=1)
        ])

        self.n_snapshots += 1
        self.snapshots = self.snapshots + [snapshot]
        self.times = np.concatenate([self.times, np.asarray([time])])
        self.mu = mu_next
        self.Sigma = Sigma_next
        return evicted

    def play_best_pair(self, n_games: int) -> None:
        i, j = best_pairing(
            self.anchor_rating,
            self.mu,
            self.Sigma,
            self.kappa
        )

        snapshot_i = self.snapshots[i] if i >= 0 else self.anchor_snapshot
        snapshot_j = self.snapshots[j]
        i_score_ij, _ = self.play_fn(
            snapshot_i, snapshot_j, n_games, self.random_key())
        _, i_score_ji = self.play_fn(
            snapshot_j, snapshot_i, n_games, self.random_key())
        self.n_played += n_games * 2

        games = [(i, j, i_score_ij + i_score_ji, n_games * 2)]
        self.mu, self.Sigma, n_sweeps = ep_update(
            self.mu,
            self.Sigma,
            self.anchor_rating,
            games,
            c=self.kappa,
        )
        if n_sweeps >= 100:
            print("Warning: EP did not converge after 100 sweeps.")

    def play(self, n_pairs: int, n_games: int) -> None:
        if self.is_empty():
            return
        for i in range(n_pairs):
            self.play_best_pair(n_games)

    def play_until(self, target_variance: npt.ArrayLike, n_games: int) -> None:
        if self.is_empty():
            return
        while self.Sigma[-1, -1] > target_variance:
            self.play_best_pair(n_games)
