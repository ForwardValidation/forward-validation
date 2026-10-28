from VolModel.GARCHModel import GARCHModel
from VolModel.EWMAModel import EWMAModel
from VolModel.GJRGARCHModel import GJRGARCHModel
from VolModel.CAViaRModel import CAViaRVolModel
from Distribution.NormalDistribution import NormalDistribution
from Distribution.StudentsDistribution import StudentsDistribution
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
import time
import os

# Folder where the figures are saved
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

def boxplot_test(test_array, y_name, config_name):
    """
    Generate and save a boxplot of the test results.

    Parameters
    ----------
    test_array : list or np.ndarray
        Array-like object containing the p-values for different estimation
        methods (e.g., MLE and Forward Validation).
    y_name : str
        Name of the variable being plotted (e.g., 'VaR violations').
    config_name : str
        Name of the model configuration, used to create the output directory
        and the plot title.
    """

    output_dir = os.path.join(RESULTS_DIR, config_name)
    os.makedirs(output_dir, exist_ok=True)
    title = f'{y_name} {config_name}'
    file_name = f'{title}.eps'
    
    plt.figure()
    for j in [0, 1]:
        plt.boxplot(test_array, tick_labels=['MLE', 'FV'])
        plt.title(title)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=10)
        plt.savefig(os.path.join(output_dir, file_name), format='eps', dpi=300)
    plt.show()
    
def backtest(test_array, conf_level):
    """
    Compute the proportion of observations below a given confidence level.

    Parameters
    ----------
    test_array : list of list or np.ndarray
        A 2D structure where each element corresponds to a series of test
        statistics (e.g., violations) for different estimation methods.
        The first dimension is assumed to index methods (e.g., 0=MLE, 1=FV).
    conf_level : float
        Confidence level for the backtest (e.g., 0.01 for 1% VaR).

    Returns
    -------
    results : list of float
        List containing the proportion of observations below the confidence
        level for each method.
    """

    results = []
    for j in [0, 1]:
        pass_test = float(round(sum(np.array(test_array[j]) < conf_level) / len(test_array[j]), 2))
        results.append(pass_test)
        
    return results

def dm_summary(losses, dm_pvals, conf_level):
    """
    Summarize the Diebold-Mariano tests between MLE and FV over all assets.

    Parameters
    ----------
    losses : list of list
        Mean losses for each asset. The first dimension indexes methods
        (0=MLE, 1=FV).
    dm_pvals : list of float
        p-values of the Diebold-Mariano test for each asset.
    conf_level : float
        Significance level of the test (e.g., 0.05).

    Returns
    -------
    results : list of float
        Proportion of assets for which the loss of FV is significantly
        lower than the loss of MLE, and proportion for which it is
        significantly higher.
    """
    mle_loss, fv_loss = np.array(losses[0]), np.array(losses[1])
    significant = np.array(dm_pvals) < conf_level
    fv_better = float(round(np.mean(significant & (fv_loss < mle_loss)), 2))
    mle_better = float(round(np.mean(significant & (fv_loss > mle_loss)), 2))

    return [fv_better, mle_better]
    
# =============================================================================
# 
# =============================================================================

# Import data ang generate the return series
inpPath = os.path.join(os.path.dirname(__file__), '..', 'data', 'daily_returns.xlsx')
retTbl = utils.prepare_data(inpPath, returns='pct')
asset_names = retTbl.columns

# Model configuration
VaR_level = 0.025
dist = NormalDistribution()
model = GJRGARCHModel(dist)
# CAViaR (specs: 'SAV', 'AS', 'IG', 'AD')
# model = CAViaRVolModel('SAV', VaR_level)
config = f'{model.config_name()} {str(VaR_level)}'

# Initialize containers for backtesting results and execution times
# Each of kupiec, indep, duration holds two sublists: 
#   - index 0 → results for MLE
#   - index 1 → results for Forward Validation (FV)
kupiec, indep, duration, dq = [[], []], [[], []], [[], []], [[], []]
extime_mle, extime_fv = [], []
# Mean Lopez and pinball losses (same indexing as above) and p-values of
# the Diebold-Mariano tests between MLE and FV for each loss
lopez, pinball = [[], []], [[], []]
dm_lopez, dm_pinball = [], []

for asset in asset_names:
    print(asset)
    returns = np.array(retTbl[asset])
    # Train Test split
    train, test = utils.train_test_split(returns, 0.8)

    # Fit using both estimation methods and store the parameters
    start_time = time.time()
    model.fit_mle(train, show=False)
    extime_mle.append(time.time() - start_time)
    params_mle = model.mle_params
    VaR_mle = model.get_valueatrisk(test, params_mle, VaR_level)
    
    start_time = time.time()
    model.fit_fv(train, VaR_level, show=False)
    extime_fv.append(time.time() - start_time)
    params_fv = model.fv_params
    VaR_fv = model.get_valueatrisk(test, params_fv, VaR_level)
    
    # Store the VaRs
    VaRs = [VaR_mle, VaR_fv]
    
    # Backtests for each VaR
    for j in [0, 1]:
        kupiec[j].append(utils.test_kupiec(test, VaRs[j], VaR_level))
        indep[j].append(utils.test_independence(test, VaRs[j], VaR_level)[0])
        duration[j].append(utils.test_duration(test, VaRs[j]))
        dq[j].append(utils.test_dq(test, VaRs[j], 4, 1, VaR_level))

    # Loss functions for each VaR
    lopez_losses = [utils.lopez_loss(test, VaRs[j]) for j in [0, 1]]
    pinball_losses = [utils.pinball_loss(test, VaRs[j], VaR_level) for j in [0, 1]]
    for j in [0, 1]:
        lopez[j].append(np.mean(lopez_losses[j]))
        pinball[j].append(np.mean(pinball_losses[j]))

    # Diebold-Mariano tests between MLE and FV for each loss
    dm_lopez.append(utils.dm_test(lopez_losses[0], lopez_losses[1]))
    dm_pinball.append(utils.dm_test(pinball_losses[0], pinball_losses[1]))

# boxplots for the p-values of the CC test
boxplot_test(indep, 'Conditional Coverage Test', config)

# Rejection rates for each test and execution time
print('Kupiec')
print('0.05', backtest(kupiec, 0.05))
print('Christoffersen')
print('0.05', backtest(indep, 0.05))
print('Duration')
print('0.05', backtest(duration, 0.05))
print('DQ')
print('0.05', backtest(dq, 0.05))
print('Ex Time')
print(f'{np.mean(extime_mle):.3f}', f'{np.mean(extime_fv):.3f}')

# Losses and Diebold-Mariano tests for each asset
loss_table = pd.DataFrame({'Lopez MLE': lopez[0],
                           'Lopez FV': lopez[1],
                           'DM Lopez p-value': dm_lopez,
                           'Pinball MLE': pinball[0],
                           'Pinball FV': pinball[1],
                           'DM Pinball p-value': dm_pinball},
                          index=asset_names)
print(loss_table.round(4).to_string())
loss_table.to_csv(os.path.join(RESULTS_DIR, config, f'Losses {config}.csv'))

# Average losses and proportion of assets where each method is significantly
# better according to the DM test: [FV better, MLE better]
print('Lopez')
print('mean', [float(round(np.mean(lopez[j]), 4)) for j in [0, 1]])
print('DM 0.05', dm_summary(lopez, dm_lopez, 0.05))
print('Pinball')
print('mean', [float(round(np.mean(pinball[j]), 4)) for j in [0, 1]])
print('DM 0.05', dm_summary(pinball, dm_pinball, 0.05))


