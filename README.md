<meta charset="UTF-8">
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
This is a trajectory planning project using receding horizon style(or MPC, if you are used to call it) using global optimization method based on IGO and GMM as searching distribution. 
Written by the first author of MGIGO, it's easy to figure out the algorithm given by the article. 

These codes are just for trials as numerical examples that try to verify whether a cost function is correct. 
Also, the codes included in MGIGO are solvers equipped with different types.

If you want to figure out the algorithm in the article MGIGO you can just use these codes. For practicle engineering uses, see https://github.com/qlp71/IOC_AGV.git

Here are formula for all algorithms:
# MGIGO: Mixture Gaussian Information Geometry Optimization

A unified information-geometric optimization framework for black-box optimization, trajectory planning, and multi-agent games.

## 1. Original MGIGO

The algorithm optimizes over $\mathbb{Z}$ using a $K$-component Gaussian mixture search distribution:

$$
p_{\Theta}(z) = \sum_{i=1}^{K} \pi_i \mathcal{N}(z; \mu_i, S_i^{-1})
$$

### Update Equations

**Component weight for sample $b$ ($i = 1,\dots,K$, $b = 1,\dots,B$):**

$$
a_{i,b}^{t} = \frac{\mathcal{N}(z_b;\mu_i^t,(S_i^t)^{-1})}{\sum_{k=1}^{K} \pi_k^t \mathcal{N}(z_b;\mu_k^t,(S_k^t)^{-1})}
$$

**Selection weights ($B_0 = \lceil a B \rceil$):**

$$
\hat{w}_b = \begin{cases}
\frac{1}{B}, & \text{if } \mathrm{rank}(f(z_b)) \le B_0 \\
0, & \text{otherwise}
\end{cases}
$$

**Mixture weights ($i = 1,\dots,K-1$):**

$$
\log\frac{\pi_{i}^{t+1}}{\pi_{K}^{t+1}} = \log\frac{\pi_{i}^{t}}{\pi_{K}^{t}} + \alpha_t \sum_{b=1}^{B} \hat{w}_b (a_{i,b}^t - a_{K,b}^t)
$$

**Covariance ($i = 1,\dots,K$):**

$$
S_{i}^{t+1} = S_{i}^t - \alpha_t \sum_{b=1}^{B} \hat{w}_b a_{i,b}^t \left(S_{i}^t (z_b - \mu_i^t)(z_b - \mu_i^t)^\top S_{i}^t - S_{i}^t\right)
$$

**Mean ($i = 1,\dots,K$):**

$$
\mu_{i}^{t+1} = \mu_{i}^t + \alpha_t (S_{i}^{t+1})^{-1} \sum_{b=1}^{B} \hat{w}_b a_{i,b}^t S_{i}^t (z_b - \mu_i^t)
$$

---

## 2. Blockwise MGIGO

For product search space $\mathbb{Z} = \mathbb{Z}_1 \times \cdots \times \mathbb{Z}_N$:

$$
p_\Theta(z) = \prod_{j=1}^N p_{\theta_j}(z^{(j)}), \qquad p_{\theta_j}(z^{(j)}) = \sum_{k=1}^K \pi_{j,k} \mathcal{N}(z^{(j)}; \mu_{j,k}, S_{j,k}^{-1})
$$

### Key Property

The selection weights $\widehat{W}_{\Theta^{t}}^{f}(z_b)^{(j)}$ are identical for all blocks $j$:

$$
\widehat{W}_{\Theta^{t}}^{f}(z_b)^{(1)} = \widehat{W}_{\Theta^{t}}^{f}(z_b)^{(2)} = \cdots = \widehat{W}_{\Theta^{t}}^{f}(z_b)^{(N)}
$$

Each block updates independently using the same global weights $\hat{w}_b$ from ranking $f(z_1,\dots,z_N)$.

### Reset Step

To prevent premature convergence, reset mixture weights every $T_0$ iterations:

$$
(\pi_{j,1},\dots,\pi_{j,K}) = \left(\frac{1}{K},\dots,\frac{1}{K}\right)
$$

---

## 3. Multi-Agent Games (Nash Equilibrium)

For $N$ agents with individual costs $f_i(z_i,z_{-i})$, define marginal expected cost:

$$
m_i(z_i; \theta_{-i}) := \mathbb{E}_{z_{-i} \sim p_{\theta_{-i}}}[f_i(z_i, z_{-i})]
$$

A randomized Nash equilibrium satisfies:

$$
E_{z_i \sim p_i^*}[m_i(z_i; \theta_{-i}^*)] \leq E_{z_i \sim p_i}[m_i(z_i; \theta_{-i}^*)], \quad \forall p_i
$$

The coupled IGO flow for each agent:

$$
\frac{d\theta_i^t}{dt} = \tilde\nabla_{\theta_i} \left.\int W_{\Theta^t}^{m_i}(z) \log p(z_i;\theta_i)\right|_{\theta_i=\theta_i^t} p_{\Theta^t}(z) dz
$$

Each agent uses its own weights $\hat{w}_{i,b}$ from Monte Carlo estimation of $m_i$, with parallel updates across agents.

### Algorithm Structure

For each agent $i$:
- Sample $z_b$ from its own distribution $p_{\theta_i}$
- Sample opponent strategies $c_m$ from opponent distributions $p_{\theta_{-i}}$
- Estimate marginal cost: $\hat{f}_i(z_b^{(i)}) = \frac{1}{M} \sum_{m=1}^M f_i(z_b^{(i)}, c_m^{(-i)})$
- Rank samples by $\hat{f}_i$ and assign weights $\hat{w}_{i,b}$
- Update $\pi_{i,k}$, $\mu_{i,k}$, $S_{i,k}$ using blockwise MGIGO updates

---

## 4. Recycling Old Samples

Importance weight for reusing sample from iteration $t-1$:

$$
\omega_b = \frac{p(z_b^{t-1}; \Lambda^{t})}{p(z_b^{t-1}; \Lambda^{t-1})} = \frac{\sum_{i=1}^K \pi_i^{t} \mathcal{N}(z_b^{t-1}; \mu_i^{t}, (S_i^{t})^{-1})}{\sum_{i=1}^K \pi_i^{t-1} \mathcal{N}(z_b^{t-1}; \mu_i^{t-1}, (S_i^{t-1})^{-1})}
$$

Quantile estimator for combined samples:

$$
\widehat{q}_{\Lambda^{t}}^{f}(z_{(k)}) = \frac{\sum_{j=1}^{k-1} \omega_{(j)}}{\sum_{j=1}^{N} \omega_{(j)}}
$$

Update equations use effective weights $\omega_b \cdot \widehat{W}_b$ for each sample.

---

