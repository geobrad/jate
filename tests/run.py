import sys
import numpy as np
import gpjax as gpx
import jax
import jax.numpy as jnp
import jate


LOG_2_PI = np.log(2.0 * np.pi)

def mock_training_process(elo_0, elo_max, mean, variance, lengthscale, steps, n, rng_key):
    prior = gpx.gps.Prior(
        kernel=gpx.kernels.Matern32(
            lengthscale=lengthscale,
            variance=variance,
        ),
        mean_function=gpx.mean_functions.Constant(mean),
    )
    t = np.arange(steps, dtype=float)
    pred = prior.predict(t[:, None])
    raw_increments = pred.sample(
        key=rng_key,
        sample_shape=(n,),
    )
    raw_trajectories = jnp.concatenate(
        [
            jnp.zeros((n, 1)),
            jnp.cumsum(raw_increments, axis=1),
        ],
        axis=1,
    )
    elo = jnp.tanh(raw_trajectories) * (elo_max - elo_0) + elo_0
    return elo


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    n_runs = int(sys.argv[1])
    n_steps = 1000
    seed = 0
    mock_key, jate_key_next = jax.random.split(jax.random.key(seed))
    elo_true = mock_training_process(
        elo_0=0,
        elo_max=4000,
        mean=0.005,
        variance=0.0001,
        lengthscale=10,
        steps=n_steps,
        n=n_runs,
        rng_key=mock_key,
    )
    mean_played = 0.0
    mean_log_likelihood = 0.0
    mean_z_score = 0.0
    mean_z2_score = 0.0
    for i in range(n_runs):
        jate_key, jate_key_next = jax.random.split(jate_key_next)

        @jax.jit(static_argnames=("n",))
        def mock_play(t_a, t_b, n, rng_key):
            r_a, r_b = elo_true[i, (jnp.asarray(t_a, dtype=int), jnp.asarray(t_b, dtype=int))]
            delta = r_a - r_b
            prob_win_a = jax.nn.sigmoid(delta * (jnp.log(10) / 400))
            scores_a = jax.random.bernoulli(rng_key, prob_win_a, (n,))
            total_score_a = jnp.sum(scores_a)
            total_score_b = n - total_score_a
            return total_score_a, total_score_b

        def rng_split_fn(rng_key) -> tuple[jax.Array, jax.Array]:
            keys = jax.random.split(rng_key)
            return keys[0], keys[1]

        jate_state = jate.init(
            max_snapshots=20,
            kappa=np.log(10.0) / 400.0,
            snapshot_zero=np.array(0.0, dtype=np.float64),
            snapshot_zero_rating=0.0,
            play_fn=mock_play,
            snapshot_variance=1000**2,
            rng_key_split_fn=rng_split_fn,
            rng_key=jate_key,
        )

        # n_steps = 10
        for t in range(1, n_steps):
            jate_state = jate.push(jate_state, t, t)
            # jate_state = jate.play(jate_state, 10, 1000)
            jate_state = jate.play_until(jate_state, 10**2, 1000)
            print(t)

        t = n_steps - 1
        times, mu, var = jate.state.belief(jate_state)
        k = np.argwhere(times == t)[0, 0]
        # errors = mu - elo_true[i, times]
        errors = mu - elo_true[i][times.astype(int)]
        # print(errors)
        likelihoods = np.exp(-np.square(errors) / (2.0 * var)) \
            / (2.0 * np.pi * var)
        log_likelihoods = -0.5 * (LOG_2_PI + np.log(var) + np.square(errors) / var)
        print(
            f"{i:3d}",
            f"{jate_state.n_played:10d}",
            f"{elo_true[i, t]:5.0f}",
            f"{errors[k]:+4.0f}",
            f"{mu[k]:5.0f}±{np.sqrt(var[k]):<4.0f}",
            f"{likelihoods[k]:8.6f}",
        )

        z_scores = errors / np.sqrt(var)

        mean_played += (jate_state.n_played - mean_played) / (i + 1)
        mean_log_likelihood += (log_likelihoods[k] - mean_log_likelihood) / (i + 1)
        mean_z_score += (z_scores[k] - mean_z_score) / (i + 1)
        mean_z2_score += (np.square(z_scores[k]) - mean_z2_score) / (i + 1)


    print(f"{mean_played:.0f}", mean_log_likelihood, mean_z_score, mean_z2_score)
