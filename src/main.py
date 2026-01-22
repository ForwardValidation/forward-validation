from VolModel.GARCHModel import GARCHModel
from VolModel.EWMAModel import EWMAModel
from VolModel.GJRGARCHModel import GJRGARCHModel
from Distribution.NormalDistribution import NormalDistribution
from Distribution.StudentsDistribution import StudentsDistribution
import numpy as np
import matplotlib.pyplot as plt
import utils
import time
import os

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

    output_dir = f'C:\\Users\\Lucas\\Desktop\\PhD\\VaR\\Results\\{config_name}\\'
    os.makedirs(output_dir, exist_ok=True)
    title = f'{y_name} {config_name}'
    file_name = f'{title}.eps'
    
    plt.figure()
    for j in [0, 1]:
        plt.boxplot(test_array, tick_labels=['MLE', 'FV'])
        plt.title(title)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=10)
        plt.savefig(output_dir + file_name, format='eps', dpi=300)
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
    
# =============================================================================
# 
# =============================================================================

# Import data ang generate the return series
inpPath = 'C:\Users\Lucas\Desktop\PhD\VaR\Data'
retTbl = utils.prepare_data(inpPath, returns='pct')
asset_names = retTbl.columns

# Model configuration
VaR_level = 0.025
dist = NormalDistribution()
model = GJRGARCHModel(dist)
config = f'{model.config_name()} {str(VaR_level)}'

# Initialize containers for backtesting results and execution times
# Each of kupiec, indep, duration holds two sublists: 
#   - index 0 → results for MLE
#   - index 1 → results for Forward Validation (FV)
kupiec, indep, duration, dq = [[], []], [[], []], [[], []], [[], []]
extime_mle, extime_fv = [], []

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


