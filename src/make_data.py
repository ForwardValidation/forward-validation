import os
import numpy as np
import pandas as pd

# Structure of the generated dataset
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'daily_returns.xlsx')
SHEET_NAME = 'inpAllFle'
START_DATE = '1995-01-02'
END_DATE = '2024-11-04'
SEED = 42
ASSET_NAMES = ['SPX Index', 'DJI Index', 'CCMP Index', 'NKY Index', 'SX5E Index',
               'DAX Index', 'CAC Index', 'FTSEMIB Index', 'IBEX Index', 'UKX Index',
               'SPTSX Index', 'ASX Index', 'NZSE Index', 'HSI Index', 'SHCOMP Index',
               'SENSEX Index', 'IBOV Index', 'MEXBOL Index', 'JALSH Index', 'WIG Index',
               'DXY Curncy', 'JPY Curncy', 'EUR Curncy', 'GBP Curncy', 'CAD Curncy',
               'AUD Curncy', 'NZD Curncy', 'HKD Curncy', 'CNH Curncy', 'INR Curncy',
               'BRL Curncy', 'MXN Curncy', 'ZAR Curncy', 'PLN Curncy', 'CL1 Comdty',
               'CO1 Comdty', 'NG1 Comdty', 'XAU Curncy', 'XAG Curncy', 'PL1 Comdty',
               'HG1 Comdty', 'LA1 Comdty', 'LX1 Comdty', 'S 1 Comdty', 'W 1 Comdty',
               'C 1 Comdty', 'CT1 Comdty']

# Range of the unconditional daily volatility (in %) drawn for each asset class
VOL_RANGES = {'Index': (0.8, 1.8),
              'Curncy': (0.3, 1.0),
              'Comdty': (1.0, 2.5)}


def draw_garch_params(rng, asset_name):
    """
    Draw random GARCH(1,1) parameters for a given asset.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    asset_name : str
        Name of the asset, used to pick a realistic volatility range
        according to its class (Index, Curncy, Comdty).

    Returns
    -------
    tuple of float
        (omega, alpha, beta) with alpha + beta < 1 (covariance stationarity).

    Notes
    -----
    omega is derived from a randomly drawn unconditional variance so that
    the simulated returns have a realistic scale (returns are in %).
    """
    vol_range = (0.5, 2.0)
    for asset_type, bounds in VOL_RANGES.items():
        if asset_type in asset_name:
            vol_range = bounds

    alpha = rng.uniform(0.03, 0.15)
    beta = rng.uniform(0.80, 0.985 - alpha)
    uncond_var = rng.uniform(*vol_range) ** 2
    omega = uncond_var * (1 - alpha - beta)

    return omega, alpha, beta


def simulate_garch(rng, n_obs, omega, alpha, beta, burn_in=500):
    """
    Simulate a GARCH(1,1) return series with Gaussian innovations.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    n_obs : int
        Number of returns to output.
    omega, alpha, beta : float
        GARCH(1,1) parameters.
    burn_in : int, optional, default=500
        Number of initial observations discarded to remove the effect of
        the starting value.

    Returns
    -------
    np.ndarray
        Simulated returns (in %) of shape (n_obs,).
    """
    n_total = n_obs + burn_in
    z = rng.standard_normal(n_total)
    returns = np.zeros(n_total)
    # initiate the variance process at the unconditional variance
    variance = omega / (1 - alpha - beta)

    for t in range(n_total):
        returns[t] = np.sqrt(variance) * z[t]
        variance = omega + alpha * returns[t] ** 2 + beta * variance

    return returns[burn_in:]


def make_mock_data(output_path=OUTPUT_PATH, seed=SEED):
    """
    Create a synthetic price dataset.

    Each asset is a price path generated from a GARCH(1,1) with random
    parameters, observed on business days between START_DATE and END_DATE.
    The asset names and sheet name follow the structure expected by
    `utils.prepare_data`.

    Parameters
    ----------
    output_path : str, optional, default=OUTPUT_PATH
        Path of the Excel file to create.
    seed : int, optional, default=SEED
        Seed of the random number generator.

    Returns
    -------
    pandas.DataFrame
        True GARCH parameters used for each asset.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(START_DATE, END_DATE, name='Date')

    mockTbl = pd.DataFrame(index=dates, columns=ASSET_NAMES, dtype=float)
    true_params = []

    for asset in ASSET_NAMES:
        omega, alpha, beta = draw_garch_params(rng, asset)
        returns = simulate_garch(rng, len(dates), omega, alpha, beta)
        # prices start at 100, returns are in %
        mockTbl[asset] = 100 * np.cumprod(1 + returns / 100)
        true_params.append([asset, omega, alpha, beta])

    true_params = pd.DataFrame(true_params, columns=['asset', 'omega', 'alpha', 'beta'])

    with pd.ExcelWriter(output_path) as writer:
        mockTbl.to_excel(writer, sheet_name=SHEET_NAME)
        true_params.to_excel(writer, sheet_name='garch_params', index=False)

    return true_params


if __name__ == '__main__':
    params = make_mock_data()
    print(params.to_string(index=False))
    print(f'Mock dataset saved to {os.path.abspath(OUTPUT_PATH)}')
