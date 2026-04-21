---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
authors:
  - name: Alan Lujan
    url: https://github.com/alanlujan91
description: >-
  The Brock-Mirman stochastic growth model and the Prescott Real Business Cycle
  model: closed-form derivations, log-linear dynamics, calibration, and the
  hours volatility puzzle.
keywords:
  - DSGE
  - Brock-Mirman
  - real business cycle
  - stochastic growth
  - hours volatility
tags:
  - dsge
  - computational
---

# Dynamic Stochastic General Equilibrium: Brock-Mirman and the Prescott RBC Model

US real GDP fluctuates with a standard deviation of about $1.76$ percent per
quarter; hours worked fluctuate with a standard deviation of $1.67$ percent.
Where do these numbers come from, and why does the canonical representative-agent
model we are about to write down generate only half of the hours volatility?
That question anchors the rest of the notebook.

This notebook is a computational companion to the DSGE lecture (Module 13). It
develops two canonical entry points to dynamic stochastic general equilibrium:
the Brock-Mirman {cite}`brockmirman:growth` stochastic growth model, the rare
DSGE model with a closed-form consumption rule, and the Prescott
{cite}`prescottTheoryAhead` Real Business Cycle model that opened the
calibration program in macroeconomics.

The notebook proceeds in four parts. Part I derives and verifies the
Brock-Mirman closed-form consumption rule using SymPy. Part II analyzes the
log-linear dynamics, simulates impulse responses, and characterizes the
stochastic steady state. Part III sets up the Prescott RBC model, derives the
intratemporal first-order condition, and calibrates the leisure weight from
long-run time use. Part IV reproduces Prescott's moment-matching table and
documents the labor volatility puzzle that motivated the New Keynesian
literature.

:::{seealso}
The [Endogenous Growth notebook](endogenous_growth.md) covers the Rebelo, Romer,
and Lucas models. The [Neoclassical Growth notebook](neoclassical_growth.md)
develops the Ramsey-Cass-Koopmans framework that underlies both models studied
here.
:::

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize, stats
from sympy import (
    symbols, Function, Rational, simplify, solve, diff, log, exp,
    Eq, init_printing, latex, expand, cancel, summation, oo,
)

init_printing()

# Theme colors
ACCENT = '#107895'
RED    = '#9a2515'
GREY   = '#5a5a5a'
```

## Model Parameters

We define two `namedtuple` types, one per model. Brock-Mirman has only three
structural parameters; Prescott RBC adds preferences over leisure, a labor
share, and a productivity process.

```{code-cell} ipython3
from collections import namedtuple

BrockMirman = namedtuple(
    'BrockMirman',
    ['alpha', 'beta', 'sigma_eps'],
)

RBCModel = namedtuple(
    'RBCModel',
    ['alpha_L', 'beta', 'rho', 'zeta', 'delta', 'g_a', 'sigma_eps'],
)

bm = BrockMirman(
    alpha=0.36,        # capital share in Cobb-Douglas
    beta=0.96,         # annual discount factor
    sigma_eps=0.0076,  # quarterly std. dev. of TFP shock (decimal form)
)

rbc = RBCModel(
    alpha_L=0.64,      # labor share (Prescott's calibration)
    beta=0.96,         # annual discount factor
    rho=1.0,           # log utility
    zeta=2/3,          # leisure weight (calibrated from time use)
    delta=0.10,        # annual depreciation
    g_a=0.004,         # mean TFP growth (quarterly)
    sigma_eps=0.0076,  # quarterly std. dev. of TFP shock
)

print("Brock-Mirman parameters:")
print(pd.Series(bm._asdict()))
print("\nPrescott RBC parameters:")
print(pd.Series(rbc._asdict()))
```

# Part I: The Brock-Mirman Closed Form

## The Social Planner's Problem

A representative agent maximizes expected discounted log utility subject to a
resource constraint with full depreciation:

```{math}
:label: dsge_bm_planner
\max \; \mathbb{E}_0 \left[\sum_{n=0}^{\infty} \beta^n \log C_{t+n}\right]
\quad \text{subject to} \quad
K_{t+1} = Y_t - C_t, \qquad
Y_{t+1} = A_{t+1} K_{t+1}^{\alpha}.
```

The capital stock depreciates 100 percent within the period. This collapses the
two state variables $(K_t, A_t)$ into a single state $Y_t$, since the household
cares only about income today, not how that income decomposes into capital and
productivity.

## The Bellman Equation and Euler Equation

Writing the problem recursively in income $Y_t$ gives

```{math}
:label: dsge_bm_bellman
v(Y_t) = \max_{C_t} \; \log C_t + \beta \, \mathbb{E}_t [v(Y_{t+1})].
```

The expectation operator $\mathbb{E}_t[\cdot]$ is the conceptual novelty
relative to the deterministic Ramsey-Cass-Koopmans problem studied in Module
11: the continuation value now integrates over all possible realizations of
$A_{t+1}$, rather than taking one deterministic path. Handling this integral
is what will force us into conjecture-and-verify below.

Letting $\mathsf{R}^k_{t+1} \equiv \alpha A_{t+1} K_{t+1}^{\alpha-1}$ denote the
gross marginal product of capital, the first-order condition gives the
Euler equation

```{math}
:label: dsge_bm_euler
1 = \beta \, \mathbb{E}_t \left[\mathsf{R}^k_{t+1} \cdot \frac{C_t}{C_{t+1}}\right].
```

This is the standard consumption Euler equation with $\mathsf{R}^k$ playing the
role of the gross interest factor. The special structure of Brock-Mirman comes
from the budget constraint $K_{t+1} = Y_t - C_t$, not from the Euler equation
itself.

## Conjecture and Verify: The MPC

Conjecture a linear consumption rule $C_t = \kappa Y_t$, where $\kappa$ is the
marginal propensity to consume. Then capital evolves
as $K_{t+1} = (1 - \kappa) Y_t$. We use SymPy to find the value of $\kappa$
that satisfies the Euler equation for every realization of $A_{t+1}$.

```{code-cell} ipython3
# Symbolic setup
alpha, beta_s, kappa, Y_t, A_tp1 = symbols(
    'alpha beta kappa Y_t A_{t+1}', positive=True
)

# Consumption rule and implied capital
C_t   = kappa * Y_t
K_tp1 = (1 - kappa) * Y_t

# Next-period income and consumption
Y_tp1 = A_tp1 * K_tp1**alpha
C_tp1 = kappa * Y_tp1

# Marginal product of capital tomorrow
R_k = alpha * A_tp1 * K_tp1**(alpha - 1)

# Euler equation right-hand side
euler_rhs = beta_s * R_k * C_t / C_tp1
euler_simplified = simplify(euler_rhs)
print(f"E[Euler RHS] = {euler_simplified}")
```

The expression simplifies to a constant in $\kappa$. The Euler equation
$1 = \text{RHS}$ then becomes a single algebraic equation in $\kappa$, which
SymPy solves directly.

```{code-cell} ipython3
# Solve for kappa
kappa_solution = solve(Eq(euler_simplified, 1), kappa)[0]
print(f"Closed-form MPC: κ = {kappa_solution}")
print(f"Equivalently:    κ = 1 - α·β  ✓")
```

The MPC is $\kappa = 1 - \alpha \beta$. The household saves exactly the fraction
The product $\alpha \beta$ equals the capital share (which governs how much tomorrow's income
rises with saving) times the discount factor (which governs patience). The
$A_{t+1}$ terms cancel exactly inside the expectation, so the rule satisfies
the Euler equation realization by realization, not only on average.

## Numerical Calibration

For our baseline parameters $(\alpha = 0.36, \beta = 0.96)$, the implied MPC
and saving rate are:

```{code-cell} ipython3
mpc_bm = 1 - bm.alpha * bm.beta
sav_bm = bm.alpha * bm.beta

calibration = pd.DataFrame({
    'Parameter': ['α (capital share)', 'β (discount factor)',
                  'κ = 1 - αβ (MPC)', 's = αβ (saving rate)'],
    'Type':  ['Input', 'Input', 'Derived', 'Derived'],
    'Value': [bm.alpha, bm.beta, mpc_bm, sav_bm],
})
calibration
```

A 65.4 percent MPC sits in the empirically plausible range for annual data, but
the implication that the saving rate equals capital's share times the discount
factor is much sharper than the data support. This is the price of the closed
form: the saving rate cannot vary with anything other than the two structural
parameters.

## The Consumption Function

```{code-cell} ipython3
Y_grid = np.linspace(0.5, 5.0, 100)

fig, ax = plt.subplots(figsize=(7, 4))
for kappa_val, color, lbl in [
    (mpc_bm, ACCENT, rf'Brock-Mirman: $\kappa = 1 - \alpha\beta = {mpc_bm:.3f}$'),
    (1.0, RED, r'Hand-to-mouth: $\kappa = 1$'),
    (0.5, GREY, r'Reference: $\kappa = 0.5$'),
]:
    ax.plot(Y_grid, kappa_val * Y_grid, lw=2, color=color, label=lbl)

ax.set_xlabel(r'Income $Y_t$')
ax.set_ylabel(r'Consumption $C_t$')
ax.set_title('The Brock-Mirman consumption rule is linear in income')
ax.grid(True, alpha=0.3)
ax.legend(frameon=False, fontsize=9)
plt.tight_layout()
plt.show()
```

# Part II: Dynamics and the Stochastic Steady State

## Log-Linear Law of Motion

Substituting the consumption rule into the resource constraint and taking logs
(letting lowercase letters denote log levels) gives an exact log-linear law of
motion for capital:

```{math}
:label: dsge_bm_lom
k_{t+1} = \log(\alpha\beta) + a_t + \alpha k_t.
```

Log capital is an AR(1) process with autoregressive coefficient equal to
capital's share $\alpha$, driven by the productivity shock $a_t$. Log output
inherits the same structure: $y_t = a_t + \alpha k_t$.

## Stochastic Steady State

Three definitions of a "stochastic steady state" appear in the literature:

1. The no-shocks limit (the value to which the economy drifts when $\varepsilon_s = 0$ for all future $s$, even though agents expect shocks)
2. The ergodic mean (the mean of $K_t$ in the stationary distribution)
3. The expected-self point ($\bar{K}$ such that $\mathbb{E}_t[K_{t+1}] = K_t$ when $K_t = \bar{K}$)

For Brock-Mirman, definition 3 coincides with definition 1. Definition 2
coincides with definitions 1 and 3 only when the shock process is itself
stationary; under a random walk in $a_t$, the ergodic distribution does not
exist.

The expected-self steady state, conditional on the current shock $a_t$, solves

```{math}
:label: dsge_bm_ss
\bar{k}(a_t) = \frac{\log(\alpha\beta) + a_t}{1 - \alpha}.
```

```{code-cell} ipython3
def stochastic_ss(model, a_t=0.0):
    """Expected-self (definition 3) stochastic steady state for log capital.

    Solves 1 = βα A_t K̄^(α-1) for K̄ when K_{t+1} = K_t = K̄. For Brock-Mirman
    this also coincides with the no-shocks steady state (definition 1).
    """
    return (np.log(model.alpha * model.beta) + a_t) / (1 - model.alpha)

k_bar = stochastic_ss(bm, a_t=0.0)
K_bar = np.exp(k_bar)
Y_bar = np.exp(0.0) * K_bar**bm.alpha
C_bar = (1 - bm.alpha * bm.beta) * Y_bar

print(f"Stochastic steady state (a_t = 0):")
print(f"  log K̄ = {k_bar:.4f}")
print(f"      K̄ = {K_bar:.4f}")
print(f"      Ȳ = {Y_bar:.4f}")
print(f"      C̄ = {C_bar:.4f}")
```

We can also verify the result with `scipy.optimize.brentq`, treating the
steady-state condition as a root-finding problem.

```{code-cell} ipython3
def ss_residual(K, model, a_t=0.0):
    """Residual of K_{t+1} = K_t when K = K̄."""
    A_t = np.exp(a_t)
    Y_t = A_t * K**model.alpha
    return model.alpha * model.beta * Y_t - K

K_bar_root = optimize.brentq(ss_residual, 0.01, 10.0, args=(bm,))
assert np.isclose(K_bar_root, K_bar), "Root finder disagrees with closed form"
print(f"brentq solution: K̄ = {K_bar_root:.6f}")
print(f"Closed form:     K̄ = {K_bar:.6f}")
```

## Impulse Response: Random Walk vs White Noise

The same model produces qualitatively different dynamics under a permanent
(random walk) shock versus a transitory (white noise) shock. We simulate both
in deviations from the no-shocks steady state.

```{code-cell} ipython3
def simulate_irf(model, shock_kind, T=30, t_shock=5):
    """Simulate log-deviation paths from a unit productivity shock at t_shock.

    Parameters
    ----------
    model : BrockMirman
    shock_kind : str
        'random_walk' (permanent) or 'white_noise' (transitory)
    T : int
        Number of periods to simulate
    t_shock : int
        Period of the shock

    Returns
    -------
    DataFrame with columns ['a', 'k', 'y', 'c'] of log deviations from
    steady state.
    """
    a = np.zeros(T)
    k = np.zeros(T)

    if shock_kind == 'random_walk':
        a[t_shock:] = 1.0
    elif shock_kind == 'white_noise':
        a[t_shock] = 1.0
    else:
        raise ValueError(f"Unknown shock_kind: {shock_kind}")

    # Law of motion in log deviations: k_{t+1} = α k_t + a_t
    for t in range(T - 1):
        k[t + 1] = model.alpha * k[t] + a[t]

    y = a + model.alpha * k
    c = y  # since C_t = κY_t and κ is constant, log deviations are equal
    return pd.DataFrame({'a': a, 'k': k, 'y': y, 'c': c})

irf_rw = simulate_irf(bm, 'random_walk')
irf_wn = simulate_irf(bm, 'white_noise')

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

for ax, df, title in [
    (axes[0], irf_rw, 'Random walk shock (permanent)'),
    (axes[1], irf_wn, 'White noise shock (transitory)'),
]:
    t = np.arange(len(df))
    ax.plot(t, df['a'], lw=2, color=GREY,   label=r'Productivity $a_t$')
    ax.plot(t, df['k'], lw=2, color=ACCENT, label=r'Capital $k_t$')
    ax.plot(t, df['y'], lw=2, color=RED,    label=r'Output $y_t$')
    ax.axvline(5, color='k', ls=':', lw=1, alpha=0.5)
    ax.set_xlabel('Period $t$')
    ax.set_ylabel('Log deviation from steady state')
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

The two panels make a sharp pedagogical point. Under a permanent shock, output
does not jump immediately to its new long-run level; capital must accumulate
first, and the AR(1) coefficient $\alpha = 0.36$ controls how slowly that
happens. Under a transitory shock, the propagation through capital outlives the
shock itself, generating serially correlated output movements from a serially
uncorrelated impulse.

## Monte Carlo: The Ergodic Distribution

When productivity follows a stationary process, log capital has a stationary
distribution we can simulate. We use a white-noise shock (so the ergodic
distribution exists) and simulate a long sample.

```{code-cell} ipython3
def simulate_path(model, T=10_000, seed=42):
    """Simulate a long path of (a_t, k_t, y_t) with white-noise shocks."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, model.sigma_eps, size=T)

    a = eps  # white noise around zero mean
    k = np.zeros(T)
    k[0] = stochastic_ss(model, a_t=0.0)
    for t in range(T - 1):
        k[t + 1] = np.log(model.alpha * model.beta) + a[t] + model.alpha * k[t]
    y = a + model.alpha * k
    return pd.DataFrame({'a': a, 'k': k, 'y': y})

path = simulate_path(bm, T=20_000, seed=42)

k_post_burn = path['k'].iloc[1000:]

# Since the law of motion k_{t+1} = log(αβ) + a_t + α k_t is linear and a_t is
# Gaussian, the stationary distribution of k is also Gaussian. Use scipy.stats
# to overlay the theoretical density.
k_mu, k_sd = k_post_burn.mean(), k_post_burn.std()
k_grid = np.linspace(k_post_burn.min(), k_post_burn.max(), 300)
k_density = stats.norm.pdf(k_grid, loc=k_mu, scale=k_sd)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(k_post_burn, bins=60, color=ACCENT, alpha=0.7,
             density=True, edgecolor='white', label='Simulated')
axes[0].plot(k_grid, k_density, color='k', lw=2, label='Gaussian fit')
axes[0].axvline(stochastic_ss(bm), color=RED, ls='--', lw=2,
                label=r'No-shocks $\bar{k}$')
axes[0].axvline(k_mu, color=GREY, ls=':', lw=2, label='Ergodic mean')
axes[0].set_xlabel(r'Log capital $k_t$')
axes[0].set_ylabel('Density')
axes[0].set_title('Ergodic distribution of $k$')
axes[0].legend(frameon=False, fontsize=9)
axes[0].grid(True, alpha=0.3)

axes[1].plot(path['k'].iloc[:200], lw=1, color=ACCENT)
axes[1].axhline(stochastic_ss(bm), color=RED, ls='--', lw=2,
                label=r'No-shocks $\bar{k}$')
axes[1].set_xlabel('Period $t$')
axes[1].set_ylabel(r'Log capital $k_t$')
axes[1].set_title('A 200-period sample path')
axes[1].legend(frameon=False, fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nMonte Carlo summary (white noise shock, T=20,000):")
print(f"  No-shocks steady state k̄        = {stochastic_ss(bm):.4f}")
print(f"  Ergodic mean of k              = {path['k'].iloc[1000:].mean():.4f}")
print(f"  Ergodic std. dev. of k         = {path['k'].iloc[1000:].std():.4f}")
print(f"  Ergodic std. dev. of y         = {path['y'].iloc[1000:].std():.4f}")
```

The ergodic mean of $\log K$ coincides with the no-shocks log steady state to
within Monte Carlo error, confirming the special tractability of the
linear-in-income consumption rule. (In levels, Jensen's inequality implies
$\mathbb{E}[K] > \exp(\mathbb{E}[\log K])$, so the level ergodic mean sits
slightly above the no-shocks level steady state.) Under a random-walk shock
instead of white noise, no stationary distribution exists, and the model
drifts indefinitely.

# Part III: The Prescott Real Business Cycle Model

## The Representative Household

Prescott {cite}`prescottTheoryAhead` introduces the canonical RBC setup. The
household maximizes

```{math}
:label: dsge_rbc_household
\mathbb{E}_0 \left[\sum_{t=0}^{\infty} \beta^t \, u(c_t, \ell_t)\right]
\quad \text{subject to} \quad
n_t + \ell_t = 1,
```

where $n_t$ is labor hours and $\ell_t$ is leisure. The endowment of time in
each period is normalized to one. The household now makes two choices each
period: how much to consume and how much to work.

## Cobb-Douglas Preferences

Prescott uses a CRRA transformation of a Cobb-Douglas aggregate of consumption
and leisure:

```{math}
:label: dsge_rbc_utility
u(c, \ell) = \frac{(c^{1-\zeta} \ell^{\zeta})^{1-\rho}}{1 - \rho},
\qquad
\lim_{\rho \to 1} u(c, \ell) = (1-\zeta) \log c + \zeta \log \ell.
```

The weight $\zeta$ governs the leisure share in the aggregator. The Cobb-Douglas
form is not arbitrary: it is the unique class of preferences that keeps the
budget share of leisure constant when wages change. Over the past century,
real wages have risen roughly fivefold while hours per worker have barely
moved {cite}`rameyFrancisLeisure`, an empirical fact only Cobb-Douglas
preferences can rationalize.

## The Intratemporal First-Order Condition

Within a period, the household chooses $c$ and $\ell$ to maximize utility
subject to the spending constraint $c + w \ell \leq \chi$, where $\chi$ is the
total expenditure on the consumption-leisure bundle. We derive the
intratemporal first-order condition symbolically.

```{code-cell} ipython3
c_s, ell_s, w_s, zeta_s, rho_s, chi_s = symbols(
    'c ell w zeta rho chi', positive=True
)

# Utility (general CRRA case)
u_general = (c_s**(1 - zeta_s) * ell_s**zeta_s)**(1 - rho_s) / (1 - rho_s)

# Marginal utilities
u_c = diff(u_general, c_s)
u_ell = diff(u_general, ell_s)

# Marginal rate of substitution = wage
mrs = simplify(u_ell / u_c)
print(f"MRS = u_ℓ / u_c = {mrs}")
```

```{code-cell} ipython3
# Setting MRS = w and solving for the wage-leisure-consumption ratio
intratemporal = Eq(mrs, w_s)
print(f"FOC: {intratemporal}")

# Multiplying through by ell/c shows that w·ℓ/c is constant
ratio = simplify(w_s * ell_s / c_s)  # what we want to express
solution = solve(intratemporal, w_s)[0]
print(f"\nSolved for w: w = {solution}")
print(f"Therefore w·ℓ/c = {simplify(solution * ell_s / c_s)}")
```

The intratemporal FOC reduces to

```{math}
:label: dsge_rbc_intra
\frac{w_t \ell_t}{c_t} = \frac{\zeta}{1 - \zeta},
```

which depends on preferences alone and not on the level of resources. Leisure
spending as a share of consumption is a structural constant.

## Calibrating the Leisure Weight

If average consumption equals average labor income $c \approx w n$, then the
intratemporal FOC implies $\ell = \zeta$. The leisure weight therefore equals
the share of waking time spent at leisure. We can verify the algebra
symbolically and compute the implied $\zeta$ from time-use data.

```{code-cell} ipython3
# Symbolic derivation: substitute c = w·(1-ℓ) into the FOC
c_subs = w_s * (1 - ell_s)
foc_substituted = solution.subs(c_s, c_subs)
print(f"After substituting c = w(1-ℓ): w = {simplify(foc_substituted)}")

# Solve for ℓ in terms of ζ
ell_from_foc = solve(Eq(foc_substituted, w_s), ell_s)[0]
print(f"\nℓ as a function of ζ: ℓ = {ell_from_foc}")
```

The symbolic result confirms $\ell = \zeta$ at the calibration point. We now
turn to the data: assuming an 8-hour sleep day, what fraction of waking time do
workers spend at leisure?

```{code-cell} ipython3
time_use = pd.DataFrame({
    'Activity': ['Work (full-time)', 'Sleep', 'Other (leisure + home)'],
    'Hours per week': [40, 56, 72],  # 8 hr sleep × 7 days = 56
})
time_use['Share of week'] = time_use['Hours per week'] / 168

# ζ is computed from waking time only
waking_hours = 168 - 56
work_hours = 40
leisure_hours = waking_hours - work_hours
zeta_implied = leisure_hours / waking_hours

print(time_use.to_string(index=False))
print(f"\nWaking hours per week: {waking_hours}")
print(f"Leisure hours per week: {leisure_hours}")
print(f"Implied ζ = ℓ = {leisure_hours}/{waking_hours} = {zeta_implied:.4f}")
print(f"Prescott calibration: ζ = 2/3 = {2/3:.4f}")
```

The arithmetic delivers an implied $\zeta \approx 0.643$, close to but not
identical to Prescott's conventional calibration of $2/3 \approx 0.667$. The
$0.024$ discrepancy arises from a choice of normalization: if instead we
define leisure as the fraction of total time not spent working (including
sleep), a 40-hour week gives $(168-40)/168 = 0.762$, which is too high. The
"waking-hours-only" normalization used here is the tighter interpretation.
Prescott rounds to $\zeta = 2/3$ to match a slightly different work-week
assumption (about $37.3$ hours); the conclusions of the model are not
sensitive to this adjustment.

## The Production Side and the Leisure Euler Equation

Output is Cobb-Douglas in capital and labor:

```{math}
:label: dsge_rbc_production
Y_t = A_t K_t^{1 - \alpha_L} n_t^{\alpha_L},
\qquad
K_{t+1} = (1 - \delta) K_t + Y_t - C_t,
```

where $\alpha_L$ is labor's share in output. Note the parameterization
difference from Brock-Mirman: in the RBC convention, $\alpha_L = 0.64$ is the
*labor* share, so capital's share is $1 - \alpha_L = 0.36$, matching the
Brock-Mirman calibration.

Combining the consumption Euler equation $c_{t+1}/c_t = \beta \mathsf{R}_{t+1}$
(where $\mathsf{R}_{t+1}$ is the gross interest factor between $t$ and $t+1$)
with the intratemporal FOC across two periods gives the leisure Euler
equation. Taking logs and rearranging produces a linear approximation in log
deviations. For any variable $X$, define $\widehat{X}_{t+1} \equiv \log X_{t+1}
- \log X_t$ (the log growth rate); let $r_{t+1} \equiv \log \mathsf{R}_{t+1}$
denote the log net interest rate; and let $\theta \equiv -\log \beta$ denote
the *time preference rate* ($\theta \approx 0.04$ when $\beta = 0.96$). Then:

```{math}
:label: dsge_rbc_leisure_euler
\widehat{\ell}_{t+1} \approx -\widehat{w}_{t+1} + (r_{t+1} - \theta).
```

A warning on notation: $\theta$ is *time preference*, while $\delta = 0.10$ is
the depreciation rate. The two differ by a factor of about 2.5, so silently
substituting one for the other changes the predicted hours response
materially.

```{code-cell} ipython3
theta = -np.log(rbc.beta)
print(f"Discount factor:     β = {rbc.beta}")
print(f"Time preference:     θ = -log(β) = {theta:.4f}")
print(f"Depreciation:        δ = {rbc.delta}")
print(f"Ratio δ/θ:           {rbc.delta/theta:.2f}  (using δ here would be a sign-magnitude error)")
```

The leisure Euler equation implies that hours fluctuate in response to two
forces: wage movements ($\widehat{w}_{t+1}$) and interest rate movements
($r_{t+1}$). In the next part, we ask whether either mechanism is
quantitatively large enough to match the data.

# Part IV: The Empirical Critique

## Prescott's Moment-Matching Table

Prescott summarizes the model's performance with a three-row table comparing
the model's standard deviations to those in the US data. The leisure standard
deviation is calibrated to match (so it is not a test); output volatility is a
respectable match; hours volatility is the failure.

```{code-cell} ipython3
moments = pd.DataFrame({
    'Statistic': [r'σ_ℓ (calibrated)', r'σ_y (output)', r'σ_n (hours)'],
    'US Data':   [0.76, 1.76, 1.67],
    'RBC Model': [0.76, 1.48, 0.76],
    'Tested?':   ['No', 'Yes', 'Yes'],
})
moments['Model/Data ratio'] = moments['RBC Model'] / moments['US Data']
moments
```

The middle row is a qualified success: the model generates output volatility at
84 percent of the data. The bottom row is a disaster: hours volatility in the
model is less than half the data, even though the calibration was
deliberately designed to match the long-run trend in hours.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(8, 4))

x = np.arange(len(moments))
width = 0.35

bars_data = ax.bar(x - width/2, moments['US Data'],
                    width, color=GREY, label='US Data')
bars_model = ax.bar(x + width/2, moments['RBC Model'],
                     width, color=ACCENT, label='RBC Model')

# Color the failing bar to highlight the puzzle
bars_model[2].set_color(RED)

for bars in (bars_data, bars_model):
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.2f}',
                    xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', fontsize=9)

ax.set_ylabel('Standard deviation (percent per quarter)')
ax.set_title('Prescott RBC matches output volatility but underpredicts hours by half')
ax.set_xticks(x)
ax.set_xticklabels([r'$\sigma_\ell$', r'$\sigma_y$', r'$\sigma_n$'])
ax.legend(frameon=False)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()
```

A quick unit reconciliation: the productivity shock standard deviation in the
deck is also 0.76 percent per quarter (we stored it as `sigma_eps = 0.0076` in
decimal form). Reporting all three series in percent makes the comparison
meaningful.

```{code-cell} ipython3
print(f"σ_ε (productivity shock):   {bm.sigma_eps * 100:.2f}% per quarter")
print(f"σ_y (output, model):        {moments.loc[1, 'RBC Model']:.2f}% per quarter")
print(f"σ_y (output, data):         {moments.loc[1, 'US Data']:.2f}% per quarter")
print(f"σ_n (hours, data):          {moments.loc[2, 'US Data']:.2f}% per quarter")
```

## The Two Failed Mechanisms

The leisure Euler equation says hours can fluctuate from either wage or
interest rate movements. Each candidate mechanism fails for a different reason.

### Mechanism One: Strong Labor Supply Response to Wages

If transitory wage movements drive hours, the intertemporal elasticity of
labor supply must be large. Micro estimates put it at 0.1 to 0.5; the value
required to match macro hours volatility is much larger.

```{code-cell} ipython3
# Two framings of the labor supply elasticity gap.

# Framing A (proportional scaling). If hours volatility σ_n scales linearly
# with the labor supply elasticity, how much would we need to scale up the
# elasticity to match the data?
sigma_n_data  = moments.loc[2, 'US Data']
sigma_n_model = moments.loc[2, 'RBC Model']
gap_factor = sigma_n_data / sigma_n_model
micro_elasticity = 0.3  # Chetty et al. consensus
scaled_elasticity = micro_elasticity * gap_factor

# Framing B (Frisch target). Independent macro-econometric work estimates the
# Frisch elasticity needed to reproduce σ_n at 2 to 3; compare to the micro
# consensus of 0.3.
frisch_required_lo, frisch_required_hi = 2.0, 3.0
ratio_lo = frisch_required_lo / micro_elasticity
ratio_hi = frisch_required_hi / micro_elasticity

print(f"Framing A: proportional scaling")
print(f"  Hours volatility gap (data/model):  {gap_factor:.2f}×")
print(f"  Scaled elasticity (micro × gap):    {scaled_elasticity:.2f}")
print(f"\nFraming B: Frisch target")
print(f"  Micro consensus elasticity:         ~{micro_elasticity:.1f}")
print(f"  Required Frisch (macro literature): {frisch_required_lo:.1f} to {frisch_required_hi:.1f}")
print(f"  Ratio required/micro:               {ratio_lo:.1f}× to {ratio_hi:.1f}×")
```

Framing B is the benchmark cited most often in labor economics: the Frisch
elasticity needed to fit macro hours data exceeds plausible micro estimates
by a factor of seven to ten.

### Mechanism Two: Strong Interest Rate Response

If interest rate movements drive hours, the consumption Euler equation
(separable utility, log case) implies

```{math}
:label: dsge_rbc_consumption_comove
\frac{c_{t+1}}{c_t} = \frac{w_{t+1} \ell_{t+1}}{w_t \ell_t}.
```

Holding wages fixed, consumption and leisure must move together. Recessions
would feature high consumption and booms low consumption. The data show
strongly procyclical consumption, exactly the opposite.

```{code-cell} ipython3
# Empirical cyclicality (BEA NIPA, postwar quarterly, HP-filtered)
cyclicality = pd.DataFrame({
    'Variable': ['Output Y', 'Consumption C', 'Hours N', 'Leisure ℓ'],
    'Correlation with Y': [1.00, 0.83, 0.86, -0.86],  # ℓ = -N
    'Sign of cyclicality': ['baseline', 'procyclical', 'procyclical', 'countercyclical'],
})
cyclicality
```

Mechanism Two predicts $\text{corr}(C, Y) < 0$ when wages are fixed and the
interest rate channel dominates. The data show $\text{corr}(C, Y) = 0.83$, a
strong positive correlation. The mechanism fails by sign, not just magnitude.

### Summers' Methodological Critique

:::{important}
Summers {cite}`summers:skeptical` frames the labor volatility failure as a
symptom of a deeper methodological problem. His argument is not primarily
about which mechanism fails: it is about what *counts as evidence* for a
structural model.

Prescott's table reports two model moments ($\sigma_y = 1.48$, $\sigma_n =
0.76$) next to two data moments and invites the reader to judge the fit by
inspection. There are no standard errors, no formal test, no rejection
region. If the model had hit $\sigma_n = 1.67$ on the nose, that would be
evidence of nothing: the calibration has enough free parameters that moment
matching is a weak discipline.

Summers' position: calibration without estimation is a rhetorical exercise,
not an empirical test. A model that matches some moments and misses others
cannot be distinguished from a model that makes no predictions at all
without a formal metric and a sampling distribution. The hours puzzle
documented above is, in Summers' reading, a *visible* failure of a
methodology that routinely produces *invisible* failures.

Prescott's reply, elaborated in later work, was that calibration is a
legitimate substitute for estimation when the model is explicitly an
approximation and the parameters come from sources independent of the
business-cycle data. The debate never fully resolved; it is worth knowing
that it happened.
:::

## What This Critique Says About DSGE

Voluntary labor supply cannot generate realistic hours dynamics in the basic
RBC framework. Two paths forward emerged in the literature:

1. **Labor market frictions**: search models (Mortensen-Pissarides) and
   indivisible labor (Hansen, Rogerson) generate large hours responses without
   requiring large elasticities at the individual level.
2. **Nominal rigidities**: New Keynesian DSGE adds sticky prices and wages,
   making hours respond to demand shocks via a different mechanism entirely.

Both extensions preserve the Euler equation at the model's center; they relax
the frictionless-price assumption that underwrites Prescott's voluntary
labor-supply mechanism. Modern central bank models (Smets-Wouters, FRB/US) are
direct descendants.

## Exercises

```{exercise}
:label: ex_dsge_rho_general
Generalize the Brock-Mirman closed form to general CRRA utility. Conjecture
$C_t = \kappa Y_t$ as before and use SymPy to find the value of $\kappa$ that
satisfies the Euler equation. Show that the closed form fails for $\rho \neq 1$
unless additional restrictions are imposed.

Hint: the Euler equation becomes $1 = \beta \mathbb{E}_t [\mathsf{R}^k_{t+1} (C_t/C_{t+1})^\rho]$,
and the $A_{t+1}$ terms no longer cancel exactly under the linear conjecture.

*Solution depends on the conjecture-and-verify cell in Part I
(variables `beta_s`, `R_k`, `C_t`, `C_tp1`).*
```

```{solution-start} ex_dsge_rho_general
:class: dropdown
```

```{code-cell} ipython3
rho_s = symbols('rho', positive=True)

# General CRRA Euler RHS under the linear conjecture
euler_rhs_general = beta_s * R_k * (C_t / C_tp1)**rho_s
euler_general = simplify(euler_rhs_general)
print(f"General CRRA Euler RHS = {euler_general}")
```

```{code-cell} ipython3
# Specialize to ρ = 1 (log utility) to recover the closed form
euler_log = euler_general.subs(rho_s, 1)
print(f"At ρ = 1: {simplify(euler_log)}")
print(f"This equals 1 when α·β = 1 - κ, i.e., κ = 1 - α·β  ✓")

# At ρ ≠ 1, A_{t+1} appears to a non-trivial power
euler_rho2 = simplify(euler_general.subs(rho_s, 2))
print(f"\nAt ρ = 2: {euler_rho2}")
print("The A_{t+1} term does not cancel; the linear rule fails for ρ ≠ 1.")
```

For $\rho \neq 1$, the $A_{t+1}$ terms enter the Euler equation with exponent
$1 - \rho$: one power from $\mathsf{R}^k_{t+1} = \alpha A_{t+1} K_{t+1}^{\alpha-1}$,
minus $\rho$ powers from $C_{t+1}^{-\rho}$ (since $C_{t+1}$ is proportional to
$A_{t+1}$ under the linear conjecture). This exponent is zero only when
$\rho = 1$. The closed-form rule $\kappa = 1 - \alpha\beta$ is a knife-edge
result: it requires log utility exactly. For any other CRRA coefficient, the
rule depends on the realization of productivity and numerical methods such as
value function iteration or the endogenous grid method are needed.

```{solution-end}
```

```{exercise}
:label: ex_dsge_persistence
The Brock-Mirman model with white-noise shocks has a stationary ergodic
distribution. Estimate the autocorrelation of log capital and log output in
the simulated path and compare to the theoretical value. Show that the AR(1)
coefficient on capital equals $\alpha$ for any zero-mean, finite-variance,
i.i.d. shock distribution (the linearization is *exact* here because of full
depreciation plus log utility, so the result does not rely on small-shock
approximations).

*Solution depends on the Monte Carlo cell in Part II (variables `path`, `bm`).*
```

```{solution-start} ex_dsge_persistence
:class: dropdown
```

```{code-cell} ipython3
# Estimate AR(1) coefficient via OLS on the simulated path
def ar1_coefficient(series):
    """Estimate the AR(1) coefficient via OLS without an intercept (after demeaning)."""
    x = series - series.mean()
    return float(np.sum(x[:-1] * x[1:]) / np.sum(x[:-1]**2))

# Use post-burnin sample
burn = 1000
k_burn = path['k'].iloc[burn:].values
y_burn = path['y'].iloc[burn:].values

ar_k = ar1_coefficient(k_burn)
ar_y = ar1_coefficient(y_burn)

results = pd.DataFrame({
    'Series': ['log K', 'log Y'],
    'AR(1) estimate': [ar_k, ar_y],
    'Theoretical α': [bm.alpha, bm.alpha],
})
results
```

The estimated AR(1) coefficient on log capital matches $\alpha = 0.36$ closely.
Output inherits the same persistence because $y_t = a_t + \alpha k_t$ and the
shock has zero autocorrelation. The result holds for any zero-mean,
finite-variance, i.i.d. shock distribution, because the law of motion for $k$
is an exact log-linear recursion (no local approximation is involved). Non-i.i.d.
shocks would inject additional persistence through $a_t$ itself.

```{solution-end}
```

```{exercise}
:label: ex_dsge_calibration_zeta
Repeat the leisure-weight calibration for two alternative work weeks: 32 hours
(a hypothetical four-day week) and 50 hours (a typical white-collar
professional). Report the implied $\zeta$ and the predicted leisure share. How
sensitive is the calibration to the assumption about sleep?

*Solution is self-contained (uses only `numpy`, `pandas`, and `matplotlib`).*
```

```{solution-start} ex_dsge_calibration_zeta
:class: dropdown
```

```{code-cell} ipython3
def calibrate_zeta(work_hours, sleep_hours_per_day=8):
    waking = 168 - 7 * sleep_hours_per_day
    leisure = waking - work_hours
    return leisure / waking

scenarios = pd.DataFrame({
    'Work week (hours)': [32, 40, 50],
    'ζ (8 hr sleep)':    [calibrate_zeta(h, 8) for h in [32, 40, 50]],
    'ζ (7 hr sleep)':    [calibrate_zeta(h, 7) for h in [32, 40, 50]],
    'ζ (9 hr sleep)':    [calibrate_zeta(h, 9) for h in [32, 40, 50]],
})
scenarios
```

```{code-cell} ipython3
# Visualize sensitivity
fig, ax = plt.subplots(figsize=(7, 4))
work_grid = np.linspace(20, 60, 100)
for sleep, color in [(7, RED), (8, ACCENT), (9, GREY)]:
    z_grid = [calibrate_zeta(w, sleep) for w in work_grid]
    ax.plot(work_grid, z_grid, lw=2, color=color, label=f'{sleep} hr sleep/day')

ax.axhline(2/3, color='k', ls=':', lw=1, alpha=0.5, label=r'Prescott $\zeta = 2/3$')
ax.axvline(40, color='k', ls=':', lw=1, alpha=0.5)
ax.set_xlabel('Work hours per week')
ax.set_ylabel(r'Implied $\zeta$')
ax.set_title('Calibrated leisure weight is sensitive to both work and sleep assumptions')
ax.legend(frameon=False, fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

The calibration is more robust to the sleep assumption than to the work week.
A 32-hour week with 8 hours of sleep raises $\zeta$ to 0.71, while a 50-hour
week lowers it to 0.55. Prescott's value is anchored to the postwar standard
work week and is pulled along with any change in that norm.

```{solution-end}
```

```{exercise}
:label: ex_dsge_hours_elasticity
The labor volatility puzzle is often summarized as requiring a Frisch
elasticity of labor supply on the order of 2 or 3, while micro estimates
cluster around 0.3. Using SymPy, derive the Frisch elasticity from the log
Cobb-Douglas utility function $u(c, \ell) = (1-\zeta)\log c + \zeta \log \ell$
(holding the marginal utility of wealth $\lambda$ fixed) and show that it
equals the structural constant $\zeta/(1-\zeta)$. Evaluate at the calibration
point and compare to the empirical targets.

*Solution requires `rbc` from the parameters cell and `symbols`, `diff`,
`simplify` from SymPy.*
```

```{solution-start} ex_dsge_hours_elasticity
:class: dropdown
```

```{code-cell} ipython3
# Derive the Frisch elasticity from log Cobb-Douglas utility
lam, w_sym = symbols('lambda w', positive=True)
zeta_sym = symbols('zeta', positive=True)

# From intratemporal FOC: λ = (1-ζ)/c, λw = ζ/ℓ
# Frisch labor supply holds λ fixed: ℓ = ζ/(λw), n = 1 - ℓ
ell_frisch = zeta_sym / (lam * w_sym)
n_frisch = 1 - ell_frisch

# Frisch elasticity ε_F = ∂ log n / ∂ log w = (w/n) * ∂n/∂w
dn_dw = diff(n_frisch, w_sym)
epsilon_F = simplify(w_sym * dn_dw / n_frisch)
print(f"Raw Frisch elasticity (as function of w): {epsilon_F}")

# At equilibrium the intratemporal FOC pins ℓ = ζ, so (λw) = ζ/ℓ_eq = 1:
epsilon_F_eq = simplify(epsilon_F.subs(lam * w_sym, zeta_sym / zeta_sym))
# Equivalently, substitute the equilibrium value of ℓ:
epsilon_F_const = simplify(epsilon_F.subs(lam * w_sym, 1))
print(f"At the calibration point: ε_F = {epsilon_F_const}")
```

```{code-cell} ipython3
# Numerical evaluation at Prescott's ζ = 2/3
zeta_val = rbc.zeta
frisch_implied = zeta_val / (1 - zeta_val)

print(f"Structural Frisch elasticity: ζ/(1-ζ) = {frisch_implied:.4f}")
print(f"Micro consensus (Chetty et al.): ~0.30")
print(f"Required to match σ_n data:      ~2.00 to 3.00")
print(f"\nImplied / micro:  {frisch_implied/0.3:.1f}×")
print(f"Implied / required (midpoint):  {frisch_implied/2.5:.2f}× "
      f"({'above' if frisch_implied >= 2.5 else 'below'} the midpoint)")
```

The Cobb-Douglas Frisch elasticity is the structural constant $\zeta/(1-\zeta)$.
At $\zeta = 2/3$ it evaluates to exactly $2$, which sits at the lower edge of
the macro-required range $[2, 3]$ and is seven times the micro consensus of
$0.3$. The puzzle is not that the model has zero elasticity: it is that the
elasticity needed to fit the data is larger than any plausible estimate from
individual-level labor economics.

```{solution-end}
```

## Summary: Where to Go Next

Two models, one lesson: the representative-agent frictionless framework can
match the volatility of output but not the volatility of employment. Brock-Mirman
is a pedagogical benchmark (closed form is possible only under three knife-edge
assumptions); Prescott RBC is the starting point of a program (calibration
against long-run moments, identification of the productivity process as the
driver of business cycles). The labor volatility puzzle documented in Part IV
is what motivates the rest of DSGE: the New Keynesian literature (Smets-Wouters,
Christiano-Eichenbaum-Evans) adds sticky prices, sticky wages, and investment
adjustment costs; the search literature (Mortensen-Pissarides, Shimer) replaces
voluntary labor supply with matching frictions. Both extensions preserve the
Euler equation at the center of the model while relaxing the frictionless
price-taking assumption.

For students continuing beyond this module: the natural next step is a New
Keynesian notebook with a three-equation canonical model (IS curve, Phillips
curve, Taylor rule) built on the same Euler-equation foundation developed here.

## References

Brock and Mirman {cite}`brockmirman:growth` introduced the stochastic growth
model with closed-form consumption rule. Kydland and Prescott
{cite}`kydlandPrescottTimeToBuild` launched the calibration program with the
canonical RBC paper, refined in Prescott {cite}`prescottTheoryAhead`. Summers
{cite}`summers:skeptical` provides the most influential contemporaneous
critique, focusing both on empirical fit and on the methodological status of
calibration as a substitute for formal estimation. The labor margin facts are
documented in Ramey and Francis {cite}`rameyFrancisLeisure`.
