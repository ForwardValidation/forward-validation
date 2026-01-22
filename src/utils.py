import numpy as np
import pandas as pd
import scipy as scp

def first_difference(df, how):
    """
    Compute returns from a DataFrame of prices using either percentage change or logarithmic returns.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing asset prices. Each column represents an asset, and each row represents a time point.
    how : str
        Method for computing returns. Options are:
        - 'pct' : percentage change returns
        - 'log' : logarithmic returns
    
    Returns
    -------
    pandas.DataFrame
        DataFrame of returns with the same columns as `df` and adjusted index 
        (first row removed).
    
    Notes
    -----
    - The first row is dropped.
    - The function scales all returns by 100 for readability.
    """
    col_names = list(df.columns)
    idx = list(df.index)[1:]

    if how == 'pct':
        new_df = df.pct_change(1) * 100
        new_df.drop(new_df.index[0], axis=0, inplace=True)
    elif how == 'log':
        new_df = pd.DataFrame(np.diff(np.log(df), axis=0) * 100, columns=col_names)

    new_df.columns = col_names
    new_df.index = idx

    return new_df


def make_portfolio(data, asset_type):
    """
    Construct a simple equally-weighted portfolio from selected assets in a DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing return series for multiple assets.
    asset_type : str
        Keyword used to select which columns (assets) to include in the portfolio.
        For example, if `asset_type='Index'`, all columns containing 'Index' in their name will be included.

    Returns
    -------
    np.ndarray
        Array of returns computed as the equally-weighted average of the selected assets. 

    Notes
    -----
    - The function searches for the `asset_type` substring in column names to select assets.
    """
    temp_array = [0] * len(data)
    n_asset = 0

    for col in data.columns:
        if asset_type in col:
            n_asset += 1
            temp_array = temp_array + data[col]

    return np.array(temp_array) / n_asset


def prepare_data(folder_path, returns='pct'):
    """
    Load and preprocess financial data from an Excel file, compute returns, 
    clean missing values, and construct simple portfolios.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing the Excel file `inpAllFle_updated.xlsx`.
    returns : str, optional, default='pct'
        Type of returns to compute from price data:
        - 'pct' : percentage change returns
        - 'log' : logarithmic returns

    Returns
    -------
    pandas.DataFrame
        Preprocessed DataFrame containing asset returns and portfolio columns:
        - Original assets after cleaning
        - Portfolios: 'Index_ptfl', 'Curncy_ptfl', 'Comdty_ptfl', 'Equity_ptfl'
        Rows with missing data are removed.

    Notes
    -----
    - The Excel file `inpAllFle_updated.xlsx` is expected to have dates as the first column 
      and asset names as column headers.
    """
    # Reads the data from an Excel file and imports it as a pandas DataFrame
    inpTbl = pd.read_excel(folder_path + r'\inpAllFle_updated.xlsx',
                           header=0,
                           index_col=0,
                           na_values='null',
                           parse_dates=True)
    inpTbl.drop(['CL1 Comdty'], axis=1, inplace=True)
    retTbl = first_difference(inpTbl, how='pct')

    # remove first row
    retTbl = retTbl.iloc[1:]

    # remove any asset that have at least 1 year of NAs
    to_remove = []
    start_date = 260
    for col in retTbl.columns:
        if pd.isnull(retTbl[col].iloc[start_date]):
            to_remove.append(col)
    retTbl.drop(to_remove, axis=1, inplace=True)

    # remove remaining NAs
    retTbl.dropna(inplace=True)

    # create the portofolios
    for asset_type in ['Index', 'Curncy', 'Comdty']:
        new_col_name = asset_type + '_ptfl'
        retTbl[new_col_name] = make_portfolio(retTbl, asset_type)

    return retTbl

def weibull_pdf(a, b, D):
    """
    Compute the Weibull probability density function (PDF) for given parameters.
    
    Parameters
    ----------
    a : float or np.ndarray
        Scale parameter of the Weibull distribution (must be positive).
    b : float
        Shape parameter of the Weibull distribution (must be positive).
    D : float or np.ndarray
        Value(s) at which to evaluate the PDF (must be non-negative).
    
    Returns
    -------
    np.ndarray
        Weibull PDF evaluated at each element of D. Values of 0 are replaced
        with 1e-6 for numerical stability.
    
    Notes
    -----
    This function ensures that zero probabilities are replaced with a small
    positive number to avoid issues in log-likelihood calculations.
    """
    val = np.array(a ** b * b * D ** (b - 1) * np.exp(-(a * D) ** b))
    val[val == 0] = 1e-6

    return val

def weibull_cdf(a, b, D):
    """
    Compute the Weibull probability density function (PDF) for given parameters.
    
    Parameters
    ----------
    see weibull_pdf
    
    Returns
    -------
    np.ndarray
        Weibull CDF evaluated at each element of D. Values of 0 are replaced
        with 1e-6 for numerical stability.
    
    Notes
    -----
    This function ensures that zero probabilities are replaced with a small
    positive number to avoid issues in log-likelihood calculations.
    """
    val = 1 - np.exp(-(a * D) ** b)

    if val != 0 and val != 1:
        return val
    elif val == 1:
        return 1 - 1e-6
    else:
        return 1e-6

def test_duration(returns, VaR):
    """
    Perform the duration-based VaR backtest.
    
    This test evaluates the temporal spacing between VaR violations using 
    a duration-based likelihood approach (Christoffersen & Pelletier, 2004).
    
    Parameters
    ----------
    returns : np.ndarray
        Array of asset returns.
    VaR : np.ndarray
        Array of Value-at-Risk estimates corresponding to `returns`.
    
    Returns
    -------
    float
        p-value of the duration-based test.
    
    Notes
    -----
    This function internally computes:
    - Durations between violations (duration)
    - Negative log-likelihood under a Weibull model (duration_loglik)
    
    References
    ----------
    Christoffersen, P. and D. Pelletier (2004). Backtesting Value-at-Risk: A Duration-Based Approach. Journal of Financial Econometrics, 2, 84–108
    """
    def duration(violations):
        """
        Compute the durations between VaR violations and censoring indicators for the first and last observations.

        """
        count = 1
        D = []
        C = []

        for i in range(0, violations.shape[0]):
            if i == 0 or i == violations.shape[0] - 1:
                if violations[i] == 0:
                    C.append(1)
                else:
                    C.append(0)
            if i == violations.shape[0] - 1 and violations[i] == 0:
                D.append(count)
            if violations[i] == 0:
                count += 1
            else:
                D.append(count)
                count = 1

        return np.array(D), np.array(C)

    def duration_loglik(params, *args):
        """
        Compute the negative log-likelihood for a sequence of durations between VaR violations under a Weibull model.
        """
        b = params[0]
        D, C = args
        C_1 = C[0]
        C_NT = C[1]
        N = D.shape[0]
        a = np.power((N - C_1 - C_NT) / sum(np.power(D, b)), (1 / b))

        loglik = (C_1 * np.log(1 - weibull_cdf(a, b, D[0]))
                  + (1 - C_1) * np.log(weibull_pdf(a, b, D[0]))
                  + sum(np.log(weibull_pdf(a, b, D[1: -1])))
                  + C_NT * np.log(1 - weibull_cdf(a, b, D[-1]))
                  + (1 - C_NT) * np.log(weibull_pdf(a, b, D[-1])))

        return -loglik
    
    violations = count_violations(returns, VaR)
    
    D, C = duration(violations)

    opt = scp.optimize.minimize(duration_loglik,
                                x0=[1.0],
                                args=(D, C),
                                bounds=[(0.001, 10.0)],
                                method='SLSQP')

    b = opt.x[0]

    h0 = -duration_loglik([1], D, C)
    h1 = -duration_loglik([b], D, C)
    ratio = 2 * (h1 - h0)

    return 1 - chi2_cdf(X=ratio, k=1)

def test_independence(returns, VaR, VaR_level):
    """
    Perform the conditional coverage (independence) test for VaR violations.

    This test evaluates whether VaR violations are independently distributed
    over time using the likelihood ratio approach described in Christoffersen (1998).

    Parameters
    ----------
    returns : np.ndarray
        Array of asset returns.
    VaR : np.ndarray or float
        Corresponding Value-at-Risk estimates. Can be a single value or an array of the same length as `returns`.
    VaR_level : float
        The VaR confidence level (e.g., 0.05 for 5% VaR).

    Returns
    -------
    p_val : float
        p-value of the independence test.
    ll_ratio : float
        Likelihood ratio statistic computed as the sum of the unconditional and
        Markov log-likelihoods.

    Notes
    -----
    - Internally, the function:
        1. Counts VaR violations (`count_violations`).
        2. Computes the binomial log-likelihood (`binomial_loglik`) for unconditional coverage.
        3. Computes the Markov log-likelihood (`markov_loglik`) for temporal dependence.
        4. Combines them into a likelihood ratio and calculates the p-value using a chi-squared distribution.

    References
    ----------
    Christoffersen, P. (1998). Evaluating Interval Forecasts. International Economic Review, 39(4), 841–862.
    """
    violations = count_violations(returns, VaR)
    b_loglik = binomial_loglik(violations, VaR_level)
    m_loglik = markov_loglik(violations)
    ll_ratio = b_loglik + m_loglik
    p_val = 1 - chi2_cdf(X=ll_ratio, k=2)
    
    return p_val, ll_ratio

def count_violations(returns, VaR):
    """
    Count the number of Value-at-Risk (VaR) violations.
    
    Parameters
    ----------
    returns : np.ndarray
        Array of asset returns.
    VaR : np.ndarray or float
        Corresponding Value-at-Risk estimates. Can be a single value or an array of the same length as `returns`.
    
    Returns
    -------
    np.ndarray
        Binary array of the same shape as `returns`:
        - 1 indicates a violation (return < VaR)
        - 0 indicates no violation
        """
    return (returns < VaR).astype('int')

def chi2_pdf(X, k):
    """Compute the probability density function (PDF) of the chi-squared distribution.

    Parameters
    ----------
    X : float or np.ndarray
        Value(s) at which to evaluate the chi-squared PDF.
    k : int or float
        Degrees of freedom of the chi-squared distribution.

    Returns
    -------
    float or np.ndarray
        The PDF value(s) of the chi-squared distribution at X.
    """
    gamma = scp.special.gamma(k / 2)

    return (1 / (2 ** (k / 2) * gamma)) * X ** (k / 2 - 1) * np.exp(-X / 2)


def chi2_cdf(X, k):
    """
    Compute the cumulative distribution function (CDF) of the chi-squared distribution.

    Parameters
    ----------
    X : float or np.ndarray
        Value(s) at which to evaluate the chi-squared CDF.
    k : int or float
        Degrees of freedom of the chi-squared distribution.

    Returns
    -------
    float or np.ndarray
        The cumulative probability P(X' <= X) for a chi-squared random variable X' with k degrees of freedom.
    """
    return scp.special.gammainc(k / 2, X / 2)


def test_kupiec(returns, VaR, VaR_level):
    """
    Perform the Kupiec (1995) unconditional coverage test for VaR violations.
    
    This test evaluates whether the observed frequency of VaR violations
    matches the expected frequency given the confidence level.
    
    Parameters
    ----------
    returns : np.ndarray
        Array of asset returns.
    VaR : np.ndarray or float
        Corresponding Value-at-Risk estimates. Can be a scalar or an array of the same length as `returns`.
    VaR_level : float
        The VaR confidence level (e.g., 0.05 for 5% VaR).
    
    Returns
    -------
    float
        p-value of the Kupiec unconditional coverage test.
    
    References
    ----------
    Kupiec, P. (1995). Techniques for Verifying the Accuracy of Risk Measurement Models. 
    The Journal of Derivatives, 3(2), 73–84.
    """
    violations = count_violations(returns, VaR)
    ll_ratio = binomial_loglik(violations, VaR_level)

    return 1 - chi2_cdf(ll_ratio, k=1)


def test_dq(returns, VaR, v_lag, f_lag, VaR_level):
    """
    Perform the Dynamic Quantile (DQ) test of Engle & Manganelli (2004) for VaR violations.

    The DQ test evaluates whether VaR violations are independent over time
    and whether the VaR model captures conditional dynamics in the returns.
    It is a regression-based test using lagged violations and VaR forecasts.

    Parameters
    ----------
    returns : np.ndarray
        Array of asset returns.
    VaR : np.ndarray
        Array of Value-at-Risk estimates corresponding to `returns`.
    v_lag : int
        Number of lagged violation terms included in the regression.
    f_lag : int
        Number of lagged VaR forecast terms included in the regression.
    VaR_level : float
        The VaR confidence level (e.g., 0.05 for 5% VaR).

    Returns
    -------
    float
        p-value of the DQ test. Values close to 1 indicate that the null hypothesis 
        of correct conditional coverage is not rejected.

    Notes
    -----
    - If a numerical issue occurs (e.g., matrix inversion fails), the function
      returns a small default p-value of 1e-6.

    References
    ----------
    Engle, R., & Manganelli, S. (2004). CAViaR: Conditional Autoregressive Value at Risk
    by Regression Quantiles. Journal of Business & Economic Statistics, 22(4), 367–381.
    """
    violations = count_violations(returns, VaR)
    p, q, n = v_lag, f_lag, violations.shape[0]
    pq = max(p, q - 1)
    y = violations[pq:] - VaR_level
    x = np.zeros((n - pq, 1 + p + q))
    x[:, 0] = 1
    
    for i in range(p):  # Lagged hits
        x[:, 1 + i] = violations[pq - (i + 1):-(i + 1)]

    for j in range(q):  # Actual + lagged VaR forecast
        if j > 0:
            x[:, 1 + p + j] = VaR[pq - j:-j]
        else:
            x[:, 1 + p + j] = VaR[pq:]

    try:
        beta = np.dot(np.linalg.inv(np.dot(x.T, x)), np.dot(x.T, y))
        lr_dq = np.dot(beta, np.dot(np.dot(x.T, x), beta)) /\
            (VaR_level * (1 - VaR_level))
        p_dq = 1 - scp.stats.chi2.cdf(lr_dq, 1 + p + q)
    
    except:
        p_dq=1e-6
    return p_dq


def markov_loglik(data):

    """
    Compute the log-likelihood ratio for a first-order Markov chain of VaR violations.

    This function evaluates the independence of VaR violations over time by comparing 
    the likelihood of the observed sequence under a Markov model (dependent violations) 
    versus a Bernoulli model (independent violations).

    Parameters
    ----------
    data : np.ndarray
        Binary array indicating VaR violations:
        - 1 if a violation occurs (return < VaR)
        - 0 otherwise
        Shape should be (n_observations,).

    Returns
    -------
    float
        Log-likelihood ratio statistic
    
    Notes
    -----
    - Adds a small constant (1e-6) to counts for numerical stability.
    """
    prev_state = data[:-1]
    next_state = data[1:]
    
    pairs = 2 * prev_state + next_state
    pairs = pairs.astype('int')
    
    n_00, n_01, n_10, n_11 = np.bincount(pairs, minlength=4)
    
    # additive smoothing for numerical stability
    n_00 += 1e-6
    n_10 += 1e-6
    n_01 += 1e-6
    n_11 += 1e-6
    
    # transition probabilities
    c_00 = n_00 / (n_00 + n_01)
    c_01 = n_01 / (n_00 + n_01)
    c_10 = n_10 / (n_10 + n_11)
    c_11 = n_11 / (n_10 + n_11)
    
    pi_2 = (n_01 + n_11) / (n_00 + n_01 + n_10 + n_11)
    
    # loglikelihood ratio
    h1 = n_00 * np.log(c_00) + n_01 * np.log(c_01) + n_10 * np.log(c_10) + n_11 * np.log(c_11)
    h0 = (n_00 + n_10) * np.log(1 - pi_2) + (n_01 + n_11) * np.log(pi_2)

    return -2 * (h0 - h1)


def binomial_loglik(data, p):
    """
    Compute the log-likelihood ratio for a sequence of VaR violations under a binomial model.

    Parameters
    ----------
    data : np.ndarray
        Binary array indicating VaR violations:
        - 1 if a violation occurs (return < VaR)
        - 0 otherwise
        Shape should be (n_observations,).
    p : float
        The expected probability of a violation under the null hypothesis (e.g., 0.05 for 5% VaR).

    Returns
    -------
    float
        Log-likelihood ratio statistic. A higher value indicates a greater deviation 
        from the expected violation probability.

    Notes
    -----
    - Uses a small numerical safeguard to avoid log(0) issues.
    """
    n = data.shape[0]
    k = max(int(sum(data)), 1e-6)
    # we use the log-likelihood to avoid numerical under/over flow
    # and other floating points precision related issues
    h0 = k * np.log(p) + (n - k) * np.log(1 - p)
    h1 = k * np.log(k / n) + (n - k) * np.log(1 - (k / n))

    return -2 * (h0 - h1)

def train_test_split(returns, train_size):
    """Split a return series into train and test sets."""
    split_idx = int(train_size * returns.shape[0])
    train = returns[: split_idx]
    test = returns[split_idx:]

    return train, test