import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable
 
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import expit
 
 
# ---------------------------------------------------------------------------
# Enumerations & configuration
# ---------------------------------------------------------------------------
 
class CAViaRSpec(Enum):
    SAV = "symmetric_abs_value"
    AS  = "asymmetric_slope"
    IG  = "indirect_garch"
    AD  = "adaptive"
 
 
_N_PARAMS: dict[CAViaRSpec, int] = {
    CAViaRSpec.SAV: 3,   # β0, β1, β2
    CAViaRSpec.AS:  4,   # β0, β1, β2+, β2-
    CAViaRSpec.IG:  3,   # β0, β1, β2
    CAViaRSpec.AD:  1,   # β1
}
 
 
# ---------------------------------------------------------------------------
# CAViaR specifications  (vectorised recursions using POSITIVE VaR internally)
# ---------------------------------------------------------------------------
 
def _quantile_sav(params: np.ndarray, returns: np.ndarray, theta: float) -> np.ndarray:
    """Symmetric Absolute Value: f_t = β0 + β1*f_{t-1} + β2*|r_{t-1}|"""
    b0, b1, b2 = params
    n = len(returns)
    f = np.empty(n)
    # Initialisation cohérente avec le niveau de VaR demandé (theta)
    f[0] = -np.quantile(returns, theta)   
    for t in range(1, n):
        f[t] = b0 + b1 * f[t - 1] + b2 * abs(returns[t - 1])
    return -f 
 
 
def _quantile_as(params: np.ndarray, returns: np.ndarray, theta: float) -> np.ndarray:
    """Asymmetric Slope: f_t = β0 + β1*f_{t-1} + β2+*(r_{t-1})+ + β2-*(r_{t-1})-"""
    b0, b1, b2p, b2m = params
    n = len(returns)
    f = np.empty(n)
    f[0] = -np.quantile(returns, theta)
    for t in range(1, n):
        r_prev = returns[t - 1]
        f[t] = b0 + b1 * f[t - 1] + b2p * max(r_prev, 0.0) + b2m * abs(min(r_prev, 0.0))
    return -f
 
 
def _quantile_ig(params: np.ndarray, returns: np.ndarray, theta: float) -> np.ndarray:
    """Indirect GARCH(1,1): f_t = sqrt(β0 + β1*f_{t-1}^2 + β2*r_{t-1}^2)"""
    b0, b1, b2 = params
    n = len(returns)
    f = np.empty(n)
    f[0] = -np.quantile(returns, theta)
    for t in range(1, n):
        inner = b0 + b1 * f[t - 1] ** 2 + b2 * returns[t - 1] ** 2
        f[t] = np.sqrt(max(inner, 1e-12))
    return -f
 
 
def _quantile_ad(params: np.ndarray, returns: np.ndarray, theta: float) -> np.ndarray:
    """Adaptive: f_t = f_{t-1} + β1 * {[1 + exp(G*(r_{t-1} + f_{t-1}))]^{-1} - θ}"""
    b1 = params[0]
    G  = 10.0 
    scale = np.std(returns)
    n  = len(returns)
    f  = np.empty(n)
    f[0] = -np.quantile(returns, theta)
    for t in range(1, n):
        indicator_smooth = expit(-G * (returns[t - 1] + f[t - 1]))
        f[t] = f[t - 1] + b1 * (indicator_smooth - theta)
        if not np.isfinite(f[t]) or abs(f[t]) > 50 * scale:
            raise FloatingPointError("AD recursion diverged")
    return -f
 
 
# ---------------------------------------------------------------------------
# Regression-quantile (tick / pinball) loss
# ---------------------------------------------------------------------------
 
def rq_loss(residuals: np.ndarray, theta: float) -> float:
    """
    Koenker & Bassett (1978) quantile regression loss:
        L = Σ_t ρ_θ(u_t)  where  ρ_θ(u) = u*(θ - I(u < 0))
    """
    return float(np.sum(residuals * (theta - (residuals < 0).astype(float))))
 
 
# ---------------------------------------------------------------------------
# Objective wrappers
# ---------------------------------------------------------------------------
 
def _make_objective(
    spec: CAViaRSpec,
    returns: np.ndarray,
    theta: float,
) -> Callable[[np.ndarray], float]:
    """Return a scalar objective function for the given specification."""
 
    def objective(params: np.ndarray) -> float:
        try:
            if spec == CAViaRSpec.SAV:
                q = _quantile_sav(params, returns, theta)
            elif spec == CAViaRSpec.AS:
                q = _quantile_as(params, returns, theta)
            elif spec == CAViaRSpec.IG:
                q = _quantile_ig(params, returns, theta)
            elif spec == CAViaRSpec.AD:
                q = _quantile_ad(params, returns, theta)
            else:
                raise ValueError(f"Unknown spec: {spec}")
 
            residuals = returns - q
            return rq_loss(residuals, theta)
        except Exception:
            return 1e10
 
    return objective
 
 
# ---------------------------------------------------------------------------
# Bounds helpers
# ---------------------------------------------------------------------------
 
def _default_bounds(spec: CAViaRSpec) -> list[tuple[float, float]]:
    """Loose but sensible parameter bounds per specification."""
    if spec == CAViaRSpec.SAV:
        return [(-5.0, 5.0), (0.0, 0.9999), (0.0, 5.0)]
    elif spec == CAViaRSpec.AS:
        return [(-5.0, 5.0), (0.0, 0.9999), (0.0, 5.0), (0.0, 5.0)]
    elif spec == CAViaRSpec.IG:
        return [(0.0, 5.0), (0.0, 0.9999), (0.0, 5.0)]
    elif spec == CAViaRSpec.AD:
        return [(0.0, 2.0)]
    else:
        raise ValueError(f"Unknown spec: {spec}")
 
 
# ---------------------------------------------------------------------------
# Main estimation routine
# ---------------------------------------------------------------------------
 
def fit_caviar(
    returns: np.ndarray,
    theta: float = 0.05,
    spec: CAViaRSpec | str = CAViaRSpec.SAV,
    bounds: list[tuple[float, float]] | None = None,
    de_kwargs: dict | None = None,
    polish_method: str = "Nelder-Mead",
    seed: int = 42,
) -> np.ndarray:
    """
    Estimate a CAViaR model by minimising the regression-quantile loss.
    """
    if isinstance(spec, str):
        spec = CAViaRSpec[spec.upper()]
 
    returns = np.asarray(returns, dtype=float)
    if bounds is None:
        bounds = _default_bounds(spec)
 
    objective = _make_objective(spec, returns, theta)
 
    de_defaults: dict = dict(
        popsize=10,
        maxiter=200,
        tol=1e-6,
        mutation = 0.8,
        recombination = 0.7, 
        polish=False
    )
    if de_kwargs:
        de_defaults.update(de_kwargs)
 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        de_result = differential_evolution(objective, bounds, **de_defaults)

    polish_result = minimize(
        objective, de_result.x, method=polish_method,
        bounds=bounds if polish_method in ("L-BFGS-B", "Powell", "TNC", "Nelder-Mead") else None,
    )
    best_params = polish_result.x if polish_result.fun < de_result.fun else de_result.x

    return best_params


# ---------------------------------------------------------------------------
# Out-of-sample VaR forecast
# ---------------------------------------------------------------------------

def forecast_caviar(
    params: np.ndarray,
    spec: CAViaRSpec | str,
    train_returns: np.ndarray,
    test_returns: np.ndarray,
    theta: float = 0.05,
) -> np.ndarray:
    """
    Produce one-step-ahead VaR forecasts over `test_returns` using corrected signs.
    """
    if isinstance(spec, str):
        spec = CAViaRSpec[spec.upper()]

    full_returns = np.concatenate([train_returns, test_returns])
    n_train = len(train_returns)
    n_total = len(full_returns)

    f = np.empty(n_total)
    f[0] = -np.quantile(train_returns, theta)

    if spec == CAViaRSpec.SAV:
        b0, b1, b2 = params
        for t in range(1, n_total):
            f[t] = b0 + b1 * f[t - 1] + b2 * abs(full_returns[t - 1])
    elif spec == CAViaRSpec.AS:
        b0, b1, b2p, b2m = params
        for t in range(1, n_total):
            r_prev = full_returns[t - 1]
            f[t] = b0 + b1 * f[t - 1] + b2p * max(r_prev, 0.0) + b2m * abs(min(r_prev, 0.0))
    elif spec == CAViaRSpec.IG:
        b0, b1, b2 = params
        for t in range(1, n_total):
            inner = b0 + b1 * f[t - 1] ** 2 + b2 * full_returns[t - 1] ** 2
            f[t] = np.sqrt(max(inner, 1e-12))
    elif spec == CAViaRSpec.AD:
        b1 = params[0]
        G = 10.0  # Alignement sur la valeur de lissage de l'estimation
        for t in range(1, n_total):
            # Correction du signe ici également (- f[t-1] remplacé par + f[t-1])
            indicator_smooth = expit(-G * (full_returns[t - 1] + f[t - 1]))
            f[t] = f[t - 1] + b1 * (indicator_smooth - theta)
    else:
        raise ValueError(f"Unknown spec: {spec}")

    return -f[n_train:]


# ---------------------------------------------------------------------------
# Adapter class
# ---------------------------------------------------------------------------

class CAViaRVolModel:
    def __init__(self, spec: CAViaRSpec | str = CAViaRSpec.SAV, var_level: float = 0.05, seed: int = 42):
        self.spec = CAViaRSpec[spec.upper()] if isinstance(spec, str) else spec
        # VaR level used by fit_mle, which has the same signature as BASEModel.fit_mle
        self.var_level = var_level
        self.seed = seed
        self.mle_params: np.ndarray | None = None
        self.fv_params: np.ndarray | None = None
        self._train_returns: np.ndarray | None = None
        self._theta: float | None = None

    def config_name(self) -> str:
        return f"CAViaR_{self.spec.value}"

    def fit_mle(self, train: np.ndarray, show: bool = False) -> np.ndarray:
        self._train_returns = np.asarray(train, dtype=float)
        self._theta = self.var_level
        self.mle_params = fit_caviar(self._train_returns, theta=self.var_level, spec=self.spec, seed=self.seed)
        return self.mle_params

    def fit_fv(self, train: np.ndarray, var_level: float = 0.05, show: bool = False) -> np.ndarray:
        self._train_returns = np.asarray(train, dtype=float)
        self._theta = var_level
        self.fv_params = fit_caviar(self._train_returns, theta=var_level, spec=self.spec, seed=self.seed)
        return self.fv_params

    def get_valueatrisk(self, test: np.ndarray, params: np.ndarray, var_level: float = 0.05) -> np.ndarray:
        if self._train_returns is None:
            raise RuntimeError("Call fit_mle or fit_fv before get_valueatrisk.")
        return forecast_caviar(
            params, self.spec, self._train_returns, np.asarray(test, dtype=float), theta=var_level
        )