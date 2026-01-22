# Forward Validation Value-at-Risk 📈

> A Python toolkit for **Value at Risk (VaR) estimation, backtesting, and simulation**, developed for the paper *“Parametric Value-at-Risk: A Forward Validation Approach for Estimating Parameters”*. 
> This repository implements standard and novel VaR estimation methods, including **MLE** and a **Forward Validation (FV)** approach, and supports multiple volatility models and distributions.

---

## Table of Contents

- Project Structure

- Installation

- Classes
	- Volatility Models (`VolModel`)

	- Distributions (`Distribution`)

- Data 

- Usage \& Minimal Reproductible Example

- Backtests

---

## Project Structure

```

├─ data/ # Raw and preprocessed datasets (Excel files)
├─ src/
│ ├─ Distribution/ # Probability distributions
│ │ ├─ NormalDistribution.py
│ │ └─ StudentsDistribution.py
│ │
│ ├─ VolModel/ # Volatility models
│ │ ├─ BASEModel.py # parent class for all models
│ │ ├─ EWMAModel.py # Model classes
│ │ ├─ GARCHModel.py
│ │ └─ GJRGARCHModel.py
│ │
│ └─ utils.py # Preprocessing, backtesting, and support functions
│
├─ main.py # Empirical VaR backtesting script (used in the paper)
├─ main_simulation.py # Monte Carlo simulation of the FV estimator
└─ README.md

```

---

## Installation

1. Clone the repository:

```bash

git clone https://github.com/LucasAurouet/forward-validation.git

```

2. Install requirements

```bash

pip install -r requirements.txt

```
Dependencies include:

- numpy

- pandas

- scipy

- matplotlib

- openpyxl

---

## Classes

**Volatility Models (VolModel)**

| Model           | Description |
|-----------------|-------------|
| `EWMAModel`     | Exponentially Weighted Moving Average volatility model |
| `GARCHModel`    | Standard GARCH model |
| `GJRGARCHModel` | Asymmetric GARCH model capturing leverage effects |

Each model implements:

- model-specific conditional variance --> `get_variance()`

- optimization support --> `init_params()`, `init_bounds()`, `constraints()` 

All models inherit from `BASEModel`, which implements:

- Maximum Likelihood Estimation (MLE) --> `fit_mle()`

- Forward Validation Estimation (FV) --> `fit_fv()`

- Value at Risk computation --> `get_valueatrisk()`

**Distributions (Distribution)**

These classes define the probability distributions for residuals and implement basic calculations.

- `loglik_resid(returns, variance, params)` --> log-likelihood of standardized residuals

- `ppf(q, params)` --> quantile function for VaR calculation

- `random_draw(mu=0, std=1)` --> generates a single random draw

---

## Data

> The original dataset used in this project comes from proprietary Bloomberg data and cannot be publicly shared due to licensing restrictions. 
> To allow users to test and reproduce the workflow, the data/ folder contains a small mock dataset with synthetic returns. 
> This mock dataset preserves the structure of the original data (dates, asset columns, portfolios) and can be used to run the example scripts and minimal reproducible examples.
> Users with access to the original data can replace the mock files in data/ with the real datasets to reproduce the full results.

`utils.prepare_data(path)` handles:

- Loading Excel data

- Calculating percentage/log returns

- Cleaning missing values

- Constructing portfolios (Index, Commodity, Currency)

- Monte Carlo Simulation (main_simulation.py)

---

## Usage

> The repository contains two full scripts, main.py and main_simulation.py, which reproduce the analyses presented in the manuscript. 
> These scripts perform empirical VaR backtesting on the proprietary dataset and run Monte Carlo simulations to evaluate the Forward Validation estimator. 
> While they are of primary interest for the paper, users without access to the original data can instead follow the Minimal Reproducible Example (MRE), which demonstrates the full workflow

### Minimal Reproducible Example

This example shows how to run a simple VaR backtest using a small synthetic dataset.  

```python

import numpy as np

from src.VolModel.EWMAModel import EWMAModel

from src.Distribution.NormalDistribution import NormalDistribution

from src import utils

# Generate synthetic returns

np.random.seed(42)

returns = np.random.normal(0, 0.01, 100)  # 100 days of returns

# Split into train and test sets

train, test = utils.train_test_split(returns, train_size=0.8)

# Initialize model and distribution

dist = NormalDistribution()

model = EWMAModel(dist)

# Estimate the parameters using FV

VaR_level = 0.05

model.fit_fv(train, VaR_level, show=True)

# Compute 5% VaR on test set

VaR = model.get_valueatrisk(test, model.fv_params, VaR_level)

# Run a simple backtest

kupiec_p = utils.test_kupiec(test, VaR, 0.05)

print(f"5% VaR Kupiec test p-value: {kupiec_p:.4f}")

```
---

## Backtests (utils.py)

> The repository includes several statistical backtests to evaluate the accuracy of VaR forecasts.
> These tests measure whether the predicted risk levels match the realized returns, both in terms of frequency of violations and their temporal dependence.
> These backtests are implemented in utils.py and can be applied to any model/distribution combination.

- `count_violations(returns, VaR)` – number of VaR violations 

- `test_kupiec(returns, VaR, VaR_level)` – unconditional coverage test

- `test_independence(returns, VaR, VaR_level)` – conditional coverage test

- `test_duration(returns, VaR)` – duration-based test

- `test_dq(returns, VaR, v_lag, f_lag, VaR_level)` – Engle & Manganelli dynamic quantile test
