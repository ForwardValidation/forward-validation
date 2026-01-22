import numpy as np
import scipy as scp

class StudentsDistribution():
    """
    Student's t distribution for modeling standardized residuals.

    Attributes
    ----------
    df : float
        Degrees of freedom of the Student-t distribution.
    """

    def __init__(self, df=None):
        self.df = df
    
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

        Notes
        -----
        Residuals are standardized by dividing returns by the conditional 
        standard deviation. A scaling factor ensures unit variance of the 
        Student-t distribution.
        """
        
        self.df = params[-1]

        std_resid = returns / np.sqrt(variance)
        x = np.sqrt(self.df / (self.df - 2)) * std_resid
        return np.log(np.sqrt(self.df / (self.df - 2))) + scp.stats.t.logpdf(x, df=self.df)
    
    def ppf(self, q, params):
        """
        Compute the percent-point function (inverse CDF) for given quantiles.

        Parameters
        ----------
        q : float or np.ndarray
            Quantile level(s) (e.g., 0.01 for 1% VaR).
        params : list or np.ndarray
            Model parameters, where the last element is the degrees of freedom.

        Returns
        -------
        float or np.ndarray
            Quantile value(s), adjusted for unit variance.
        """
        
        self.df = params[-1]

        return scp.stats.t.ppf(q, df=self.df) / np.sqrt(self.df / (self.df - 2))
    
    def random_draw(self, mu=0.0, std=1.0):
        """
        Generate a random draw from the Student-t distribution.

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
        return scp.stats.t.rvs(df=self.df, loc=mu, scale=std, size=1)[0] / np.sqrt(self.df / (self.df - 2))