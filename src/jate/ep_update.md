# Expectation Propagation update for a Gaussian rating belief

Mathematical formulation implemented by `ep_update.py`.

The function takes a Gaussian belief over snapshot ratings and a batch of new game results, and returns a Gaussian approximation of the Bayes posterior, matched in mean and covariance (moment matching).

---

## 1. Problem

**Belief.** Ratings $r \in \mathbb{R}^n$, one entry per tracked snapshot. The prior is

$$
p_0(r) = \mathcal{N}(r;\, \mu_0, \Sigma_0), \qquad \Lambda_0 = \Sigma_0^{-1}, \qquad \eta_0 = \Lambda_0 \mu_0 ,
$$

where $\Lambda_0$ is the precision and $\eta_0$ the natural mean. $\Sigma_0$ must be positive definite.

**Data.** A batch of $K$ pairs, indexed $k = 1,\dots,K$, one per unordered pair that played. Pair $k$ is $(i_k, j_k)$ with

- $N_k$ = number of games played,
- $W_k \in [0, N_k]$ = total score of $i_k$ against $j_k$ (win $=1$, draw $=\tfrac12$, loss $=0$).

**Observation model.**

$$
P(i \text{ beats } j) = \sigma\big(c\,(r_i - r_j)\big), \qquad \sigma(s) = \frac{1}{1 + e^{-s}},
$$

with $c = 1$ if ratings are in logit units and $c = \ln 10 / 400$ if they are in Elo points.

Define the projection direction and scalar for pair $k$:

$$
u_k = c\,(e_{i_k} - e_{j_k}), \qquad s_k = u_k^\top r .
$$

The likelihood of pair $k$ depends on $r$ only through $s_k$:

$$
f_k(s) = \sigma(s)^{W_k}\, \sigma(-s)^{N_k - W_k} .
$$

This is the binomial likelihood without the constant binomial coefficient. With draws, $W_k$ is fractional and $f_k$ is a quasi-likelihood.

**Posterior.**

$$
p(r) \;\propto\; \mathcal{N}(r;\, \mu_0, \Sigma_0)\; \prod_{k=1}^{K} f_k\!\left(u_k^\top r\right).
$$

This is not Gaussian, because the logistic likelihood is not conjugate to a Gaussian. The goal is a Gaussian $q(r) = \mathcal{N}(r; \mu, \Sigma)$ whose mean and covariance approximate those of $p(r)$.

The key structural fact is that each factor sees $r$ only through one scalar projection $s_k$. This is why every EP step below is one-dimensional plus a rank-one update.

---

## 2. EP approximating family

Each likelihood factor is replaced by an unnormalized Gaussian *site* in its own projection:

$$
t_k(s) = \exp\!\Big(-\tfrac12\, \tau_k\, s^2 + \nu_k\, s\Big),
$$

with site precision $\tau_k$ and site natural mean $\nu_k$. The approximation is

$$
q(r) \;\propto\; \mathcal{N}(r;\mu_0,\Sigma_0)\prod_{k=1}^{K} t_k\!\left(u_k^\top r\right),
$$

which is Gaussian with

$$
\Lambda = \Lambda_0 + \sum_{k} \tau_k\, u_k u_k^\top, \qquad
\eta = \eta_0 + \sum_{k} \nu_k\, u_k, \qquad
\Sigma = \Lambda^{-1}, \qquad
\mu = \Sigma\, \eta .
$$

**Initialization:** $\tau_k = \nu_k = 0$ for all $k$, so $q$ starts as the prior. Sites exist only for this batch. Earlier evidence is already inside $(\mu_0, \Sigma_0)$, so no old sites are kept.

---

## 3. The site update

One update of site $k$, with $u = u_k$ and $(\tau, \nu) = (\tau_k, \nu_k)$ its current values.

### 3.1 Marginal of the projection under $q$

A linear projection of a Gaussian is Gaussian:

$$
s = u^\top r \;\sim\; \mathcal{N}(m_s, v_s), \qquad m_s = u^\top \mu, \qquad v_s = u^\top \Sigma\, u .
$$

### 3.2 Cavity

Remove site $k$'s contribution from the projected marginal, $q^{\setminus k}(s) \propto q(s)/t_k(s)$. In natural parameters this is a subtraction:

$$
\frac{1}{v_c} = \frac{1}{v_s} - \tau, \qquad
\frac{m_c}{v_c} = \frac{m_s}{v_s} - \nu .
$$

The cavity is a valid Gaussian only if $1/v_c > 0$. If $1/v_s - \tau \le 10^{-12}$ the site is skipped for this sweep.

The $n$-dimensional cavity is never formed. Only its marginal on $s$ is needed.

### 3.3 Tilted distribution

Replace the site by the exact likelihood factor:

$$
\hat p_k(s) \;\propto\; f_k(s)\; \mathcal{N}(s;\, m_c, v_c),
$$

and let $\hat m$ and $\hat v$ be its mean and variance (computed in Section 4).

**Why one-dimensional moments suffice.** In the $n$-dimensional tilted distribution $\hat p(r) \propto q^{\setminus k}(r)\, f_k(u^\top r)$, the conditional law of $r$ given $s$ is the cavity's conditional, which is Gaussian and unchanged by $f_k$:

$$
\hat p(r) = \hat p_k(s)\; q^{\setminus k}(r \mid s).
$$

So the mean and covariance of $\hat p(r)$ depend on $f_k$ only through $\hat m$ and $\hat v$. Matching them with a Gaussian of the form (cavity $\times$ site) reduces to choosing the site so that the projected marginal has mean $\hat m$ and variance $\hat v$.

### 3.4 New site parameters

Require cavity $\times$ site to have mean $\hat m$ and variance $\hat v$ in the projection. Natural parameters add, so

$$
\tau^{\text{new}} = \max\!\left(\frac{1}{\hat v} - \frac{1}{v_c},\; 0\right), \qquad
\nu^{\text{new}} = \frac{\hat m}{\hat v} - \frac{m_c}{v_c} .
$$

The $\max(\cdot, 0)$ is a numerical safeguard. In exact arithmetic it is inactive (Section 6.1).

**Damping** with factor $\delta \in (0, 1]$ ($\delta = 1$ means none):

$$
\tau \leftarrow (1 - \delta)\,\tau + \delta\, \tau^{\text{new}}, \qquad
\nu \leftarrow (1 - \delta)\,\nu + \delta\, \nu^{\text{new}} .
$$

Let $\Delta\tau$ and $\Delta\nu$ be the changes in $\tau$ and $\nu$ produced by this step.

### 3.5 Rank-one update of the global belief

Since $q$ already contains the old site, only the change is applied:

$$
\Lambda \leftarrow \Lambda + \Delta\tau\; u u^\top, \qquad \eta \leftarrow \eta + \Delta\nu\; u .
$$

In $(\mu, \Sigma)$ form, by the Sherman–Morrison formula with $v_s = u^\top \Sigma u$:

$$
\Sigma' = \Sigma - \frac{\Delta\tau}{1 + \Delta\tau\, v_s}\, (\Sigma u)(\Sigma u)^\top ,
$$

$$
\mu' = \mu + \frac{\Delta\nu - \Delta\tau\, m_s}{1 + \Delta\tau\, v_s}\; \Sigma u .
$$

**Derivation of the mean update.** From the covariance update,

$$
\Sigma' \Lambda = I - \frac{\Delta\tau}{1 + \Delta\tau\, v_s}\, \Sigma u\, u^\top, \qquad
\Sigma' u = \frac{\Sigma u}{1 + \Delta\tau\, v_s}.
$$

Then $\mu' = \Sigma'(\eta + \Delta\nu\, u) = \Sigma'\Lambda\,\mu + \Delta\nu\, \Sigma' u$, and using $u^\top \mu = m_s$,

$$
\mu' = \mu - \frac{\Delta\tau\, m_s}{1 + \Delta\tau\, v_s}\, \Sigma u + \frac{\Delta\nu}{1 + \Delta\tau\, v_s}\, \Sigma u ,
$$

which is the stated update. Cost is $O(n^2)$ per site, dominated by the outer product.

---

## 4. Tilted moments by adaptive Gauss–Hermite quadrature

Let the log of the unnormalized tilted density be

$$
h(s) = W \log\sigma(s) + (N - W)\log\sigma(-s) - \frac{(s - m_c)^2}{2 v_c} .
$$

Its derivatives are

$$
h'(s) = \big(W - N\sigma(s)\big) - \frac{s - m_c}{v_c}, \qquad
-h''(s) = N\sigma(s)\big(1 - \sigma(s)\big) + \frac{1}{v_c} > 0 .
$$

So $h$ is strictly concave with a unique mode $\hat s$.

**Step 1: locate the tilted mode.** Newton's method on $h'$, started at $s = m_c$, with the step clipped to $\pm 3\sqrt{v_c}$, run for a fixed 30 iterations:

$$
s \leftarrow s + \operatorname{clip}\!\left(\frac{h'(s)}{-h''(s)},\; \pm 3\sqrt{v_c}\right).
$$

Set the local variance $\tilde v = 1 / \big(-h''(\hat s)\big)$.

**Step 2: quadrature centred on the tilted mode.** Substitute $s = \hat s + \sqrt{2\tilde v}\, x$, so $ds = \sqrt{2\tilde v}\, dx$:

$$
\int g(s)\, e^{h(s)}\, ds
= \sqrt{2\tilde v} \int g(s(x))\, e^{h(s(x)) + x^2}\; e^{-x^2}\, dx
\;\approx\; \sqrt{2\tilde v} \sum_{i=1}^{M} w_i\, g(s_i)\, e^{h(s_i) + x_i^2},
$$

where $(x_i, w_i)$ are the $M$-point Gauss–Hermite nodes and weights for the weight $e^{-x^2}$ (default $M = 40$), and $s_i = \hat s + \sqrt{2\tilde v}\, x_i$.

**Step 3: moments.** Self-normalized weights, computed in log space with the maximum subtracted:

$$
\omega_i = \frac{w_i\, e^{h(s_i) + x_i^2}}{\sum_j w_j\, e^{h(s_j) + x_j^2}}, \qquad
\hat m = \sum_i \omega_i s_i, \qquad
\hat v = \sum_i \omega_i\, (s_i - \hat m)^2 .
$$

**Why centre on the tilted mode.** Near $\hat s$, $h(s) \approx h(\hat s) - x^2$, so $e^{h + x^2}$ is nearly constant and the quadrature is nearly exact for a Gaussian-like tilted density. Centring on the cavity instead places the nodes with spread $\sqrt{2 v_c}$. When the cavity is wide compared with the likelihood (a new snapshot with a diffuse prior, or many games in a pair), almost no nodes land on the tilted mass and the moments are wrong.

---

## 5. Algorithm

```
input : mu0, Sigma0, games = [(i_k, j_k, W_k, N_k)], scale c, damping delta, tolerance tol
init  : mu <- mu0, Sigma <- Sigma0, tau_k <- 0, nu_k <- 0 for all k

repeat (sweep) until max_k(|d_tau_k|, |d_nu_k|) < tol over a full sweep:
    for k = 1..K:
        u      <- c (e_i - e_j)
        Su     <- Sigma u                      # O(n)
        m_s    <- u . mu ;  v_s <- u . Su
        prec_c <- 1/v_s - tau_k
        if prec_c <= 1e-12: continue           # invalid cavity: skip site
        v_c    <- 1/prec_c ;  m_c <- v_c (m_s/v_s - nu_k)
        (m_hat, v_hat) <- tilted_moments(m_c, v_c, W_k, N_k)       # Section 4
        tau_new <- max(1/v_hat - prec_c, 0)
        nu_new  <- m_hat/v_hat - m_c * prec_c
        tau_u <- (1-delta) tau_k + delta tau_new
        nu_u  <- (1-delta) nu_k  + delta nu_new
        d_tau <- tau_u - tau_k ;  d_nu <- nu_u - nu_k
        mu    <- mu + Su (d_nu - d_tau m_s) / (1 + d_tau v_s)
        Sigma <- Sigma - d_tau (Su)(Su)^T / (1 + d_tau v_s)
        tau_k <- tau_u ;  nu_k <- nu_u

return mu, (Sigma + Sigma^T)/2, number of sweeps
```

The final symmetrization removes the small asymmetry that accumulates from repeated rank-one updates.

---

## 6. Properties

### 6.1 Site precisions are non-negative

For the tilted density, $V(s) = -\log \hat p_k(s)$ has $V''(s) = N\sigma(s)(1-\sigma(s)) + 1/v_c \ge 1/v_c$. The Brascamp–Lieb variance inequality gives

$$
\operatorname{Var}_{\hat p_k}(s) \;\le\; \mathbb{E}\!\left[\frac{1}{V''(s)}\right] \;\le\; v_c ,
$$

so $\hat v \le v_c$ and $\tau^{\text{new}} = 1/\hat v - 1/v_c \ge 0$. The clamp in Section 3.4 therefore only guards against numerical error. The reason is that the logistic likelihood is log-concave.

### 6.2 Fixed point

At convergence, for every pair $k$, the mean and variance of $s_k$ under $q$ equal those of the tilted distribution $\hat p_k$. Moment consistency holds simultaneously for all sites. This is the EP fixed-point condition.

Convergence is not guaranteed in general. Damping ($\delta < 1$) and the cavity-validity skip are the safeguards.

### 6.3 Sweeps versus a single pass

A single sweep from zero sites is assumed-density filtering (ADF), and its result depends on the order of the pairs. Repeated sweeps remove that order dependence. Across batches the code only carries $(\mu, \Sigma)$, which is ADF-style carrying. Within a batch it is full EP.

### 6.4 Cost

Per site, $O(n^2)$ for the rank-one update plus $O(30 + M)$ scalar operations for the quadrature, which does not depend on $n$. Per sweep, $O(K n^2)$.

### 6.5 What is reported

The output $\mu'$ is the approximate posterior **mean**, not the mode (MAP). The uncertainty of a rating difference is

$$
\operatorname{Var}(r_i - r_j) = \Sigma_{ii} + \Sigma_{jj} - 2\,\Sigma_{ij} .
$$

---

## 7. Relation to the Laplace approximation

Laplace finds the mode $\hat r = \arg\max_r \log p(r)$ and uses the curvature there:

$$
\Sigma_{\text{L}}^{-1} = \Lambda_0 + \sum_k N_k\, \sigma(\hat s_k)\big(1 - \sigma(\hat s_k)\big)\; u_k u_k^\top, \qquad \hat s_k = u_k^\top \hat r .
$$

EP instead uses the mean and variance of each tilted distribution, rather than mode and curvature. These differ when the tilted density is skewed, which happens with sparse or lopsided results (for example 5–0). With many games per pair and a moderately informative prior the two nearly coincide.

---

## 8. Symbols and code

| Math | Code |
|---|---|
| $\mu, \Sigma$ | `mu`, `Sigma` |
| $c$ | `c` |
| $W_k, N_k$ | `W`, `N` |
| $\Sigma u$ | `Su` |
| $m_s, v_s$ | `m_s`, `v_s` |
| $1/v_c$ | `prec_c` |
| $v_c, m_c$ | `v_c`, `m_c` |
| $\hat m, \hat v$ | `m_new`, `v_new` (from `tilted_moments`) |
| $\tau_k, \nu_k$ | `tau[k]`, `nu[k]` |
| $\Delta\tau, \Delta\nu$ | `dt`, `dn` |
| $\delta$ | `damping` |
| $M$ | `order` |
| $\hat s, \tilde v$ | `s0`, `var` |

---

## 9. Scope and caveats

- **Gaussian family.** The result is a Gaussian approximation. Any skew in the true posterior is lost.
- **Draws.** Fractional $W_k$ makes $f_k$ a quasi-likelihood, not a proper binomial.
- **Proper prior required.** $\Sigma_0$ must be positive definite, so that cavities have positive precision. A prior that fixes the rating scale (for example an anchor) is assumed.
- **Verification so far.** On small test problems (4 players), the implementation:
  - matches an $O(n^3)$ reference implementation of the same EP to about $10^{-11}$,
  - matches an importance-sampling reference posterior to within Monte Carlo noise, including a diffuse prior with few games and a diffuse prior with many games per pair.
- **Not yet tested.** A JAX port, and accumulation of approximation error over many sequential carried-forward updates.
