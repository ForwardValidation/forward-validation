import numpy as np
from VolModel.BASEModel import BASEModel
from Distribution.NormalDistribution import NormalDistribution
from Distribution.StudentsDistribution import StudentsDistribution

class EWMAModel(BASEModel):
    """
    Exponentially Weighted Moving Average (EWMA) volatility model.
    This model estimates conditional variance using an exponentially
    weighted average of past squared returns, as originally popularized
    by RiskMetrics.
    The model supports both Normal and Student-t conditional distributions.
    """

    def get_variance(self, returns, params):
        """
        Compute the conditional variance process using the EWMA.

        Parameters
        ----------
        returns : np.ndarray
            Array of asset returns of shape (T,).
        params : list or np.ndarray
            Model parameters. The first element corresponds to the EWMA
            persistence parameter :beta.

        Returns
        -------
        np.ndarray
            Array of conditional variances of shape (T,).

        Notes
        -----
        The initial variance is set to the unconditional variance of the
        return series.
        """
        beta = params[0]
        alpha = 1 - beta
        # initiate the variance process
        variance = [np.var(returns)]
        for t in range(1, returns.shape[0]):
            var_t = (beta * variance[t - 1]
                     + alpha * returns[t - 1] ** 2)
            variance.append(var_t)

        return np.array(variance)
    
    def init_params(self):
        """
        Provide initial parameter values for optimization.

        Returns
        -------
        list
            Initial parameter values:
            - Normal distribution: [beta]
            - Student-t distribution: [beta, nu]
        """
        dist = self.distribution
        # starting values for the optimization algorithms 
        if isinstance(dist, NormalDistribution):
            return [0.98]
        elif isinstance(dist, StudentsDistribution):
            return [0.98, 5]
    
    def init_bounds(self):
        """
        Provide parameter bounds for optimization.
        Bounds are defined to ensure stationarity and well-defined moments.

        Returns
        -------
        list of tuple
            List of (lower, upper) bounds for each parameter.

        Raises
        ------
        ValueError
            If the distribution is not recognized.
        """
        dist = self.distribution
        # outputs bounds for the optimization algorithms 
        if isinstance(dist, NormalDistribution):
            return [(0.0, 1.0)]
        elif isinstance(dist, StudentsDistribution):
            return [(0.0, 1.0), (2.0 + 1e-6, 100)]
        else:
           raise ValueError('unknown distribution')
    
    def constraints(self, params):
        """
        Define inequality constraints for the optimization problem.
        This constraint enforces the EWMA stability condition.
        This method is designed to be used with optimization routines from
        scipy.optimize, which require constraints
        to be provided as callable functions.


        Parameters
        ----------
        params : list or np.ndarray
            Model parameters.

        Returns
        -------
        float
            Constraint value, which must be non-negative to be satisfied.
        """
        # outputs constraints for the optimization algorithms
        alpha = params[0]
        eps = 1e-6

        return 1 - alpha - eps
    
    def config_name(self):
        """
        Return a human-readable name for the model configuration.
        The name depends on the chosen conditional distribution.

        Returns
        -------
        str
            Model configuration name.
        """
        if isinstance(self.distribution, NormalDistribution):
            dist_name = 'Normal'
        elif isinstance(self.distribution, StudentsDistribution):
            dist_name = 'Student'
        return 'EWMA ' + dist_name

    

        
        
            
            
        