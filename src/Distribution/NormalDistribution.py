import numpy as np
import scipy as scp
import random

class NormalDistribution():
    
    def loglik_resid(self, returns, variance, params):
        """
        Compute the log-likelihood of standardized residuals.

        Parameters
        ----------
        returns : np.ndarray
            Array of asset returns.
        variance : np.ndarray
            Conditional variance series.
        params : list or np.ndarray
            Model parameters, where the last element corresponds to degrees of freedom.

        Returns
        -------
        np.ndarray
            Log-likelihood values for each time step.

        """
        std_resid = returns / np.sqrt(variance)
        
        return scp.stats.norm.logpdf(std_resid)
    
    def ppf(self, q, params):
        """
        Compute the percent-point function (inverse CDF) for given quantiles.

        Parameters
        ----------
        q : float or np.ndarray
            Quantile level(s) (e.g., 0.01 for 1% VaR).
        params : list or np.ndarray
            Model parameters.

        Returns
        -------
        float or np.ndarray
            Quantile value(s).
        """
        # quantile function
        return scp.stats.norm.ppf(q)
    
    def random_draw(self, mu=0, std=1):
        """
        Generate a random draw from the Normal distribution.

        Parameters
        ----------
        mu : float, default=0.0
            Location parameter (mean).
        std : float, default=1.0
            Scale parameter.

        Returns
        -------
        float
            Random draw, adjusted to have unit variance.
        """
                
        return scp.stats.norm.rvs(loc=mu, scale=std, size=1)[0]