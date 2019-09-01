import numpy as np
import scipy.linalg as scl
import sys
from numpy import ndarray
from numba import jit
from scipy import stats
from typing import Sequence, Union, Any, Optional, List, Tuple


class Regressor:
    def __init__(self, predictors: Sequence[ndarray], response: ndarray,
                 method='linear') -> None:
        method = method.lower()
        if isinstance(predictors, np.ndarray):
            self.predictors = [predictors]
        else:
            self.predictors = [predictor.flatten() for predictor in predictors]
        self.response = response.flatten()
        self.orders: Optional[List[List[int]]] = None
        self.β: Optional[List[float]] = None
        self.design: Optional[ndarray] = None

    def fit(self, orders: Sequence[Union[int, Sequence[int]]]) -> None:
        """ Perform a fitting using the exponents in orders

        Args:
            orders: The orders to use in construction of the
                design matrix. If an element is an int,
                all lower orders will be used as well.
                The constant term is always present.
        """
        for i, order in enumerate(orders):
            if not isiterable(order):
                orders[i] = list(range(1, order+1))
        self.orders = orders
        self.vandermonde = vandermonde(self.predictors, orders)
        if np.linalg.cond(self.vandermonde) < sys.float_info.epsilon:
            β = lin_reg_inv(self.vandermonde, self.response)
        else:
            β = lin_reg_svd(self.vandermonde, self.response)
        self.β = β
        return β

    def predict(self, predictors: Union[ndarray, Sequence[ndarray]]) -> ndarray:
        """ Predict the response based on fit

        Args:
            predictors: The predictors to predict from.
        Returns:
            The predicted values.
        """
        assert self.β is not None, "Perform fitting before predicting"
        if isinstance(predictors, np.ndarray):
            if len(self.predictors) != 1:
                raise ValueError("Must provide same amount of predictors")
            shape = predictors.shape
            predictors = [predictors]
        else:
            if len(predictors) != len(self.predictors):
                raise ValueError("Must provide same amount of predictors")
            shape = predictors[0].shape
            predictors = [predictor.flatten() for predictor in predictors]

        X = vandermonde(predictors, self.orders)
        y = (X * self.β).sum(axis=1)
        return y.reshape(shape)

    def r2(self, predictors: Optional[Sequence[ndarray]] = None,
           response: Optional[ndarray] = None) -> float:
        """ Calculates the R² score

        Args:
            predictors: If no predictors are provided, the
                training predictors will be used. Must be set together
                with response.
            response: If no response is provided, the
                training response will be used. Must be set together
                with predictors.
        Returns:
            The R² score

        Raises:
            AssertionError if only one of predictors or response is provided.
        """
        if predictors is not None:
            ỹ = self.predict(predictors)
            assert response is not None, "Must provide a response"
        else:
            ỹ = self.predict(self.predictors)
        if response is not None:
            y = response
            assert predictors is not None, "Must provide predictors"
        else:
            y = self.response
        y̅ = y.mean()
        return 1 - np.sum((y - ỹ)**2) / np.sum((y - y̅)**2)

    def mse(self, predictors: Optional[Sequence[ndarray]] = None,
            response: Optional[ndarray] = None) -> float:
        """ Calculates the mean square error (MSE)

        Args:
            predictors: If no predictors are provided, the
                training predictors will be used. Must be set together
                with response.
            response: If no response is provided, the
                training response will be used. Must be set together
                with predictors.
        Returns:
            The MSE

        Raises:
            AssertionError if only one of predictors or response is provided.
        """
        if predictors is not None:
            ỹ = self.predict(predictors)
            assert response is not None, "Must provide a response"
        else:
            ỹ = self.predict(self.predictors)
        if response is not None:
            y = response
            assert predictors is not None, "Must provide predictors"
        else:
            y = self.response
        return np.mean((y - ỹ)**2)

    @property
    def SSE(self) -> float:
        """ Error sum of squares """
        return np.sum((self.response - self.predict(self.predictors))**2)

    @property
    def sigma2(self) -> float:
        """ Estimate of σ² """
        N, p = self.vandermonde.shape
        # Note that N - p = N - (k+1)
        std_err = 1/(N - p) * self.SSE
        return std_err

    @property
    def var(self) -> ndarray:
        X = self.vandermonde
        N, p = X.shape
        Σ = np.linalg.inv(X.T@X)
        return np.diag(Σ)*self.sigma2

    @property
    def tscore(self) -> ndarray:
        tscore = self.β/np.sqrt(self.var)
        return tscore

    def ci(self, alpha) -> ndarray:
        """ Compute the 1-2α confidence interval

        Assumes t distribution of N df.

        Args:
            alpha: The percentile to compute
        Returns:
            The lower and upper limits of the CI as p×2 matrix
            where p is the number of predictors + intercept.
        """
        X = self.vandermonde
        N, p = X.shape
        zalpha = np.asarray(stats.t.interval(alpha, N - p))
        σ = np.sqrt(self.var)
        ci = np.zeros((p, 2))
        for i, β in enumerate(self.β):
            ci[i, :] = β + zalpha*σ[i]
        return ci


def vandermonde(predictors: Sequence[ndarray],
                orders: Sequence[Sequence[int]]) -> ndarray:
    """ Constructs a Vandermonde matrix

    Each predictor is raised to the exponents given in orders.

    Args:
        predictors: The predictors to use. Must have same equal length
        orders: Each predictor must be accompanied by a list of exponents.
    Returns:
        The resulting Vandermonde matrix
    Raises:
        ValueError or AssertionError if the input have inequal sizes.
    """
    if len(predictors) != len(orders) or not predictors:
        raise ValueError("Must provide same number of predictors as orders")

    size = predictors[0].size
    X = np.ones((size, sum(len(order) for order in orders) + 1))
    i = 1
    for predictor, order in zip(predictors, orders):
        assert predictor.size == size, "Predictors must have same size"
        for n in order:
            X[:, i] = (predictor**n).flatten()
            i += 1
    return X


@jit(nopython=True)
def lin_reg_inv(X, y):
    return np.linalg.inv(X.T@X)@X.T@y


def lin_reg_svd(x: ndarray, y: ndarray) -> ndarray:
    u, s, v = scl.svd(x)
    return v.T @ scl.pinv(scl.diagsvd(s, u.shape[0], v.shape[0])) @ u.T @ y


def isiterable(obj: Any) -> bool:
    try:
        _ = iter(obj)
        return True
    except TypeError:
        return False
