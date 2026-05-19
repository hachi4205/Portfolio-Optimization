# Portfolio Optimization on the Vietnamese Stock Market

Final project for Linear Algebra. Builds an intelligent portfolio of HOSE blue-chip stocks
using PCA, multiple linear regression, and Sharpe-ratio optimization — implemented from
scratch with NumPy and SciPy.

**Author:** Nguyen Ha Chi · Student ID: 2024106126

---

## Project overview

Given 8 Vietnamese blue-chip stocks (VNM, VIC, VHM, FPT, HPG, MWG, VCB, MBB), find
portfolio weights `w` that maximize the Sharpe ratio:

```
maximize    (μᵀw - r_f) / sqrt(wᵀΣw)
subject to  Σwᵢ = 1   (fully invested)
            wᵢ ≥ 0    (long-only)
```

Here `μ` is the vector of expected returns (estimated by MA5 or MLR), `Σ` is the
covariance matrix of daily returns, and `r_f` is the risk-free rate (4%/year, proxied by
the Vietnam 10-year government bond).

## Pipeline

```
Prices ──► Returns + Σ ──► PCA ──► μ forecast ──► Optimization ──► Evaluation
                                  (MA5 / MLR)      (Sharpe max)     (backtest)
```

| Notebook | Topic |
|---|---|
| `01_data_collection.ipynb` | Download HOSE prices via `vnstock` |
| `02_exploratory_analysis.ipynb` | Returns, covariance matrix Σ, correlations |
| `03_pca_from_scratch.ipynb` | PCA implemented manually via eigendecomposition |
| `04_moving_average_prediction.ipynb` | MA5 forecast → μ estimate |
| `05_linear_regression.ipynb` | Multiple Linear Regression on PC scores → μ estimate |
| `06_portfolio_optimization.ipynb` | SLSQP Sharpe maximization + backtest |

## Linear algebra used

| Concept | Application |
|---|---|
| Matrix-vector product | Portfolio expected return: `μᵀw` |
| Quadratic form | Portfolio variance: `wᵀΣw` |
| Eigendecomposition | PCA — eigenvalues/eigenvectors of Σ |
| Linear system solving | OLS coefficient estimation in MLR |
| Constrained optimization | Sharpe maximization on the simplex |

## Key results

**In-sample performance (Section 7 of NB06)**

| Portfolio | Annual return | Volatility | Sharpe |
|---|---:|---:|---:|
| 1/N benchmark (MLR μ) | -3.07% | 20.09% | -0.35 |
| Optimal-Historical | 20.94% | 24.70% | +0.69 |
| Optimal-MA5 | 13.00% | 23.62% | +0.38 |
| **Optimal-MLR** | **34.81%** | **22.33%** | **+1.38** |

The MLR-based optimal portfolio achieves the highest in-sample Sharpe ratio.

**Out-of-sample backtest (Section 9 of NB06)**

Train on 75% of data, hold weights fixed, apply to the remaining 25%.

| Portfolio | Total return | Sharpe | Max drawdown |
|---|---:|---:|---:|
| Optimal (trained on train set) | -14.69% | -0.77 | -30.51% |
| 1/N benchmark | +9.74% | +0.38 | -15.20% |

**The naïve 1/N benchmark beats the optimal portfolio out-of-sample.** This replicates
the well-known finding of DeMiguel, Garlappi & Uppal (2009): mean-variance optimization
is highly sensitive to estimation error in `μ` and tends to over-concentrate on past
winners that may reverse.

The gap between in-sample (+1.38 Sharpe) and out-of-sample (-0.77 Sharpe) is the
central finding of the project — it surfaces the estimation-error problem of Markowitz
optimization on Vietnamese market data.

## Repository structure

```
.
├── data/         # CSVs: prices, returns, Σ, PCA outputs, μ estimates, results
├── figures/     # Charts: weights, pie, risk-return scatter, Sharpe comparison, backtest
├── notebooks/   # 01–06 (run sequentially)
└── README.md
```

## How to run

```bash
pip install pandas numpy scipy matplotlib seaborn scikit-learn vnstock jupyter
jupyter notebook
```

Then run notebooks 01 → 06 in order. Each notebook reads from `data/` and writes its
outputs back, so later notebooks depend on earlier ones.

## Tech stack

Python 3.13 · pandas · numpy · scipy · matplotlib · seaborn · scikit-learn · vnstock

## Future work

- **Shrinkage estimators** (Ledoit-Wolf) to stabilize Σ
- **Robust optimization** with uncertainty sets around μ
- **Rolling-window backtest** with periodic re-optimization
- **Risk-parity / minimum-variance** portfolios — do not require μ, more robust
- **L2 regularization** on weights to discourage concentration

## Reference

DeMiguel, V., Garlappi, L., & Uppal, R. (2009). *Optimal Versus Naive Diversification:
How Inefficient is the 1/N Portfolio Strategy?* Review of Financial Studies, 22(5),
1915–1953.
