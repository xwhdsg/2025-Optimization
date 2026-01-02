# Topic Options (choose ONE)

## Topic 1 (Convex + Nonsmooth): Group LASSO (Row-Sparse Multi-Task Regression)

### (T1.1) Problem

Given (A\in\mathbb{R}^{m\times n}), (B\in\mathbb{R}^{m\times \ell}), (\mu>0),
[
\min_{X\in\mathbb{R}^{n\times \ell}}
;; \frac12|AX-B|*F^2 + \mu\sum*{i=1}^{n}|X_{i,:}|_2.
]
(Encourages **row sparsity**.)

### (T1.2) Data generation (fixed)

Use:

* seed = **97006855**
* n = 512, m = 256, (\ell=2), (k=\text{round}(0.1n)), (\mu = 10^{-2})
* (A \sim \mathcal{N}(0,1)^{m\times n})
* pick active row indices (p) of size (k)
* (U\in\mathbb{R}^{n\times \ell}): (U_{p,:}\sim\mathcal{N}(0,1)), others 0
* (B = AU)

### (T1.3) Required baseline (choose ≥1)

* **CVXPY** solution as reference (F^\star) (recommended)
* `torch.optim.LBFGS` or `Adam` on a smooth surrogate (optional baseline)

### (T1.4) Implement (your own update rules): choose ≥3

* Proximal gradient (ISTA) with group (row-wise) shrinkage prox
* Accelerated proximal gradient (FISTA)
* ADMM on split (X=Z)
* Subgradient method
* Block coordinate descent by rows

### (T1.5) Mandatory metrics

* objective gap to (F^\star)
* time-to-tolerance (e.g., gap ≤ 1e-3)
* convergence curves (objective vs iteration and vs time)

---

## Topic 2 (Non-Convex): Low-Rank Matrix Completion via Factorization

### (T2.1) Problem

Given observed set (\Omega\subset[n]\times[n]),
[
\min_{U,V\in\mathbb{R}^{n\times r}}
;; \frac12 |\mathcal{P}_\Omega(UV^\top - M^\star)|_F^2

* \frac{\lambda}{2}\left(|U|_F^2+|V|_F^2\right).
  ]

### (T2.2) Data generation (fixed)

* n = 300, r = 5, sampling rate p = 0.1, (\lambda=10^{-3})
* (U^\star,V^\star\sim\mathcal{N}(0,1)), (M^\star=U^\star(V^\star)^\top)
* sample (\Omega): include each entry i.i.d. with probability p
* run **at least 3 different seeds** and report mean±std

### (T2.3) Required baseline (choose ≥1)

* `torch.optim.Adam` or `torch.optim.LBFGS` on the same objective
* (bonus) convex nuclear norm formulation via CVXPY at smaller n

### (T2.4) Implement (your own update rules): choose ≥3

* Full-batch GD on ((U,V))
* Mini-batch SGD over observed entries
* Alternating minimization (least-squares subproblems)
* (optional) momentum / your own Adam-like update (do not rely on torch.optim for the “ours” version)

### (T2.5) Mandatory metrics

* recovery error (|UV^\top-M^\star|_F/|M^\star|_F)
* time-to-target recovery
* training loss vs iteration/time
* stability across seeds (mean±std)

---

## Topic 3 (Online Convex Optimization): Online Logistic Regression + Regret

### (T3.1) Online problem

At each round (t=1,\dots,T), choose (w_t) s.t. (|w_t|*2\le R), observe ((a_t,y_t)), incur
[
f_t(w)=\log(1+\exp(-y_t a_t^\top w)),\quad y_t\in{-1,+1}.
]
Regret:
[
\mathrm{Reg}*T=\sum*{t=1}^T f_t(w_t)-\min*{|w|\le R}\sum_{t=1}^T f_t(w).
]

### (T3.2) Data generation (fixed)

* d = 50, T = 5000, R = 5
* generate (w^\star), normalize to (|w^\star|_2=R/2)
* (a_t\sim\mathcal{N}(0,I))
* (y_t=\mathrm{sign}(a_t^\top w^\star+\epsilon_t)), (\epsilon_t\sim\mathcal{N}(0,0.5^2))
* (optional) non-stationary: rotate (w^\star) every 1000 rounds

### (T3.3) Required baseline (choose ≥1)

* offline comparator via **CVXPY** (recommended)
* or high-precision projected GD as a proxy offline optimum

### (T3.4) Implement (your own update rules): choose ≥2

* Online Gradient Descent (OGD) with projection onto (\ell_2) ball
* FTRL / RFTL with quadratic regularizer
* (optional) AdaGrad-style OCO

### (T3.5) Mandatory metrics

* (\mathrm{Reg}_T) vs (T) and (\mathrm{Reg}_T/T) vs (T)
* runtime (total and/or per-round)
* (optional) online prediction accuracy vs time

---

## Topic 4 (Bandits): Stochastic + Adversarial Multi-Armed Bandits

### (T4.1) Problem

K = 10 arms, horizon T = 20000. At time (t), pick arm (I_t), observe reward only for that arm.

### (T4.2) Environments (fixed)

You must test **both**:

**Env A (stochastic Bernoulli):**

* best arm mean = 0.60
* other means spaced in [0.45, 0.58]
* reward (r_t\sim \mathrm{Bernoulli}(\mu_{I_t}))

**Env B (choose one):**

* non-stationary: best arm switches every 4000 steps
  **or**
* adversarial: periodic switching reward patterns

### (T4.3) Required baseline (choose ≥1)

* a tuned (\epsilon)-greedy baseline (simple baseline)
* or any available library baseline you can cite (optional)

### (T4.4) Implement (your own update rules): choose ≥2

* UCB1
* Thompson Sampling
* EXP3
* (optional) discounted/sliding-window UCB

### (T4.5) Mandatory metrics

* regret vs time (define appropriately):

  * stochastic: (T\mu^\star-\sum_{t=1}^T r_t)
  * adversarial: regret vs best fixed arm in hindsight
* mean±std over **≥20 runs**
* runtime

---
