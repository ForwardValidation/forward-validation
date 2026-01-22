import numpy as np
import scipy as scp
import utils

class BASEModel():
    
    """
    Base class for conditional volatility models.

    This class provides a common interface for volatility models used in
    the paper.
    It implements generic estimation routines while delegating model-
    specific components (e.g. variance dynamics, constraints) to subclasses.

    Subclasses must implement at least:
    - get_variance
    - init_params
    - init_bounds
    - constraints

    The class supports:
    - Maximum Likelihood Estimation (MLE)
    - Forward Validation (FV) based on conditional coverage tests
    """
    
    def __init__(self, distribution):
        # storing variables in the object
        self.distribution = distribution
        self.returns = None

    def fit_mle(self, returns, show):
        
        """
        Estimate model parameters via Maximum Likelihood Estimation (MLE).

        Parameters
        ----------
        returns : np.ndarray
            Array of asset returns of shape (T,).
        show : bool
            If True, display optimization diagnostics.

        Returns
        -------
        np.ndarray
            Estimated parameter vector obtained by MLE.
        """
        
        self.returns = returns
        
        def neg_loglik(params):
            # compute the negative loglikelihood for a given set of parameters
            # uses the general form loglik - 0.5 * log(variance)
            variance = self.get_variance(self.returns, params)
            loglik = self.distribution.loglik_resid(self.returns, variance, params)
            
            return - np.sum(loglik - 0.5 * np.log(variance)) 
        
        opt = scp.optimize.minimize(neg_loglik,
                                    x0=self.init_params(),
                                    method='SLSQP',
                                    bounds=self.init_bounds(),
                                    constraints={'type': 'ineq', 
                                                 'fun': self.constraints
                                                 },
                                    options={'disp': show,
                                             'eps': 1e-3
                                             }
                                    )
        # store the optimal parameters
        self.mle_params = opt.x
        
        return opt.x

    def fit_fv(self, returns, VaR_level, show):
        """
        Estimate model parameters using Forward Validation (FV).

        Parameters
        ----------
        returns : np.ndarray
            Array of asset returns of shape (T,).
        VaR_level : float
            VaR confidence level (e.g. 0.01 or 0.05).
        show : bool
            If True, display intermediate optimization values.

        Returns
        -------
        np.ndarray
            Estimated parameter vector obtained via Forward Validation.
        """

        self.VaR_level = VaR_level
        self.show = show
        self.returns = returns
        
        def LRcc(params):
            # objective function
            VaR = self.get_valueatrisk(self.returns, params, self.VaR_level)
            violations = utils.count_violations(self.returns, VaR)
            b_loglik = utils.binomial_loglik(violations, self.VaR_level)
            m_loglik = utils.markov_loglik(violations)
            
            if self.show:
                print(params, b_loglik + m_loglik)

            return b_loglik + m_loglik

        opt = scp.optimize.basinhopping(LRcc,
                                        x0=self.init_params(),
                                        niter=5,
                                        T=0.01,
                                        stepsize=0.005,
                                        niter_success=5,
                                        disp=show,
                                        minimizer_kwargs={
                                            'method': 'SLSQP',
                                            'bounds': self.init_bounds(),
                                            'options': {'eps': 1e-3},
                                            'constraints': {
                                                'type': 'ineq', 
                                                'fun': self.constraints
                                                }
                                            }
                                        )
        
        self.fv_params = opt.x
        
        return opt.x
    
    def get_valueatrisk(self, returns, params, VaR_level):
        
        """
        Compute the Value-at-Risk (VaR) series.

        Parameters
        ----------
        returns : np.ndarray
            Array of asset returns of shape (T,).
        params : list or np.ndarray
            Model parameters.
        VaR_level : float
            VaR confidence level.

        Returns
        -------
        np.ndarray
            Time series of Value-at-Risk estimates.
        """
        
        variance = self.get_variance(returns, params)
        f = self.distribution.ppf(VaR_level, params)
        
        return f * np.sqrt(variance)

   
