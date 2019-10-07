import numpy as np
import scipy.linalg as scl
import itertools
from numpy import ndarray
import sys
from scipy import stats
from typing import Sequence, Union, Any, Optional, List, Tuple, Dict
from sklearn.linear_model import Lasso as skLasso
import warnings


class Regressor:
    """ Base class implementing ordinary least square regression

    Ordinary least square polynomial regression using SVD decomposition.
    Supports basic analysis features such as variance, degrees of freedom
    and confidence intervals of β.

    Attributes:
        response (ndarray): The respones variable to model
        orders (list of int): The polynomial order of each predictor
            to use in constructing the design matrix.
        β (ndarray): The coefficients once the system has been solved.
        design_matrix (ndarray): The constructed design matrix
        interactions (bool): Whether or not use use interactions between
            the polynomials.
        max_interaction (int): The maximum order of interactions to use.
            If None, no restriction is used.
        condition_number (float): The condition number of the projection matrix
    """
    def __init__(self, predictors: Sequence[ndarray],
                 response: ndarray) -> None:
        """ Initialize the class

        Args:
            predictors: A sequence of each predictor used for
                predicting the response.
            response: The response to model using the predictors
        """
        if isinstance(predictors, np.ndarray):
            self.predictors = [predictors]
        else:
            self.predictors = [predictor.flatten() for predictor in predictors]
        self.response = response.flatten()
        self.orders: Optional[Sequence[Sequence[int]]] = None
        self.β: Optional[List[float]] = None
        self.design_matrix: Optional[ndarray] = None
        self.interactions: bool = False
        self.max_interaction: Optional[int] = None
        self.condition_number = 0

    def fit(self, orders: Sequence[Union[int, Sequence[int]]],
            interactions: bool = False,
            max_interaction: Optional[int] = None) -> ndarray:
        """ Perform a fitting using the exponents in orders

        Args:
            orders: The orders to use in construction of the
                design matrix. If an element is an int,
                all lower orders will be used as well.
                The constant term is always present.
            interactions: Whether or not to use interactions
               of the polynomial degrees.
        """
        # Construct the orders to fit
        for i, order in enumerate(orders):
            if not isiterable(order):
                orders[i] = list(range(1, order+1))
        self.orders = orders

        # Construct the design matrix
        self.max_interaction = max_interaction
        if interactions or max_interaction is not None:
            self.interactions = True

        design_matrix = self.make_design_matrix(self.predictors)

        if design_matrix.shape[0] < design_matrix.shape[1]:
            warnings.warn("Number of features surpasses number of samples")

        self.design_matrix = self.standardize(design_matrix)

        self.β = self.solve(self.design_matrix, self.response)
        return self.β

    def standardize(self, matrix: ndarray) -> ndarray:
        """ Standardize the predictors of the matrix

        Shifts the mean of each column to zero and scales by the
        standard deviation.

        Also sets the `standardizer` attribute.

        Args:
            matrix: The matrix to standardize
        Returns:
            The standardized matrix.
        """
        # Standardize the matrix
        mean = matrix[:, 1:].mean(axis=0)
        std = matrix[:, 1:].std(axis=0)

        def standardizer(mat: ndarray) -> ndarray:
            # Ignore the first column of constant term
            mat[:, 1:] = (mat[:, 1:] - mean[np.newaxis, :])/std[np.newaxis, :]
            return mat

        self.standardizer = standardizer
        return standardizer(matrix)

    def make_design_matrix(self, predictors: List[ndarray]) -> ndarray:
        """ Construct the design matrix given predictors and orders

        The design matrix has the construction

            1 X X² ... X^N Y Y² ... Y^m XY X²Y ... X^NY XY² X²Y² ... X^NY^N

        Uses the attributes `orders` and `max_interaction`

        Args:
           predictors: The predictors, or basis functions, to use to
               construct the design matrix.
        Returns:
            The design matrix
        """

        matrix = vandermonde(predictors, self.orders)

        # Handle interaction terms
        if self.interactions:
            matrix = add_interactions(matrix, self.orders,
                                      self.max_interaction)
        return matrix

    def solve(self, design_matrix: ndarray, response: ndarray) -> ndarray:
        """ Solve the system using inversion or SVD

        Args:
            design_matrix: The matrix to solve.
            response: The response to model
        Returns:
            The solution β

        """
        self.condition_number = np.linalg.cond(design_matrix)
        if self.condition_number < sys.float_info.epsilon:
            β = lin_reg_inv(design_matrix, response)
        else:
            β = lin_reg_svd(design_matrix, response)
        return β

    def predict(self, predictors: Union[ndarray, Sequence[ndarray]]) -> ndarray:
        """ Predict the response based on fit

        The resulting prediction matrix is standardized using the same
        standardization applied to the design matrix

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

        X = self.make_design_matrix(predictors)
        X = self.standardizer(X)

        # If the constant coefficient is taken care of elsewhere
        if X.shape[1] == len(self.β) - 1:
            y = self.β[0] + X@self.β[1:]
        else:
            y = X@self.β
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
        N, p = self.design_matrix.shape
        # Note that N - p = N - (k+1)
        std_err = 1/(N - p) * self.SSE
        return std_err

    @property
    def var(self) -> ndarray:
        """ The variance Var(β) """
        X = self.design_matrix
        N, p = X.shape
        Σ = np.linalg.inv(X.T@X)
        return np.diag(Σ)*self.sigma2

    @property
    def tscore(self) -> ndarray:
        tscore = self.β/np.sqrt(self.var)
        return tscore

    def ci(self, alpha: float) -> ndarray:
        """ Compute the 1-2α confidence interval

        Assumes t distribution of N df.

        Args:
            alpha: The percentile to compute
        Returns:
            The lower and upper limits of the CI as p×2 matrix
            where p is the number of predictors + intercept.
        """
        X = self.design_matrix
        N, p = X.shape
        zalpha = np.asarray(stats.t.interval(alpha, N - p))
        σ = np.sqrt(self.var)
        ci = np.zeros((p, 2))
        for i, β in enumerate(self.β):
            ci[i, :] = β + zalpha*σ[i]
        return ci

    def betadict(self) -> Dict[str, float]:
        """ Get the terms of β and their coefficients """
        assert self.β is not None
        assert self.orders is not None
        coeffs = {'const': self.β[0]}
        i = 1
        for order in self.orders[0]:
            coeffs['x^'+str(order)] = self.β[i]
            i += 1

        for order in self.orders[1]:
            coeffs['y^'+str(order)] = self.β[i]
            i += 1

        if self.interactions:
            for x, y in itertools.product(*self.orders):
                if self.max_interaction is not None:
                    if x * y > self.max_interaction:
                        continue
                coeffs['x^'+str(x)+'y^'+str(y)] = self.β[i]
                i += 1

        return coeffs

    def df(self) -> ndarray:
        """ Compute the degrees of freedom as tr(H) """
        X = self.design_matrix
        assert X is not None
        H = X@np.linalg.inv(X.T@X)@X.T
        return np.trace(H)


class Ridge(Regressor):
    """ Implement Ridge regularization on top of OLS
    
    Attributes:
        Same as Regressor.
        parameter: The regularization parameter
    """
    def __init__(self, predictors: Sequence[ndarray],
                 response: ndarray,
                 parameter: float) -> None:
        """ Initialize the matrix

        Args:
            predictors: A sequence of each predictor used for
                predicting the response.
            response: The response to model using the predictors
            parameter: The regularization parameter
        """
        super().__init__(predictors, response)
        self.parameter = parameter

    def make_design_matrix(self, predictors: ndarray) -> ndarray:
        """ Construct the design matrix given predictors and orders

        Does not have a constant term

        The design matrix has the construction

            X X² ... X^N Y Y² ... Y^m XY X²Y ... X^NY XY² X²Y² ... X^NY^N

        Uses the attributes `orders` and `max_interaction`

        Args:
           predictors: The predictors, or basis functions, to use to
               construct the design matrix.
        Returns:
            The design matrix
        """

        matrix = super().make_design_matrix(predictors)
        # The constant coefficient can be removed
        matrix = matrix[:, 1:]
        return matrix

    def standardize(self, matrix: ndarray) -> ndarray:
        """ Standardize the predictors of the matrix

        Shifts the mean of each column to zero and scales by the
        standard deviation. Takes into account that the design matrix
        has to constant term.

        Also sets the `standardizer` attribute.

        Args:
            matrix: The matrix to standardize
        Returns:
            The standardized matrix.
        """
        # Standardize the matrix
        mean = matrix.mean(axis=0)
        std = matrix.(axis=0)

        def standardizer(mat):
            # Ignore the first column of constant term
            mat = (mat - mean[np.newaxis, :])/std[np.newaxis, :]
            return mat
        self.standardizer = standardizer
        return standardizer(matrix)

    def solve(self, design_matrix: ndarray, response: ndarray) -> ndarray:
        """ Solve the system using SVD

        Adds regularization along the diagonal of X^T X

        Args:
            design_matrix: The matrix to solve.
            response: The response to model
        Returns:
            The solution β
        """

        X = design_matrix
        y = response
        # if np.linalg.cond(self.design_matrix) < sys.float_info.epsilon:
        β = np.linalg.inv(X.T@X + self.parameter*np.eye(X.shape[1]))@X.T@y
        # The constant is given by 1/N Σ y_i
        β = np.array([np.mean(y), *β])
        return β

    def df(self) -> ndarray:
        X = self.design_matrix
        assert X is not None
        H = X@np.linalg.inv(X.T@X + self.parameter*np.eye(X.shape[1]))@X.T
        return np.trace(H)


class Lasso(Ridge):
    """ Implements Lasso regularizaion

    Leverages the code of Ridge, only changing how the system is solved.
    The solving is relegated to sklearn.Lasso

    """
    def solve(self, design_matrix: ndarray, response: ndarray) -> ndarray:
        """ Solve the system using sklearn.Lasso

        Adds regularization along the diagonal of X^T X

        Args:
            design_matrix: The matrix to solve.
            response: The response to model
        Returns:
            The solution β
        """
        X = design_matrix
        y = response
        clf = skLasso(alpha=self.parameter, fit_intercept=False)
        clf.fit(X, y)
        β = np.array([np.mean(y), *clf.coef_])
        return β


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


def add_interactions(vandermonde: ndarray,
                     orders: Sequence[Sequence[int]],
                     max_interaction: Optional[int] = None) -> ndarray:
    """ Add interaction terms to the vandermonde matrix

    Args:
        vandermonde: The polynomial vandermonde matrix to use as
            basis for creating the interactions.
        orders: The orders corresponding to each column in the matrix
        max_interaction: The maximum degree of the interacting polynomial
            If None, no restriction is used
    Returns:
        The complete matrix.
    """
    # First column is constant term
    assert len(orders) == 2, "Only two term interactions supported"
    offset = len(orders[0])
    for col_x, col_y in itertools.product(*orders):
        if max_interaction is not None:
            if col_x * col_y > max_interaction:
                continue
        col_y += offset
        product = vandermonde[:, col_x] * vandermonde[:, col_y]
        vandermonde = np.append(vandermonde, product[..., None], axis=1)
    return vandermonde


def lin_reg_inv(X: ndarray, y: ndarray):
    """ Solve the system Xβ = y by matrix inversion"""
    return np.linalg.inv(X.T@X)@X.T@y


def lin_reg_svd(x: ndarray, y: ndarray) -> ndarray:
    """ Solve the system Xβ = y by singular value decomposition"""
    u, s, v = scl.svd(x)
    return v.T @ scl.pinv(scl.diagsvd(s, u.shape[0], v.shape[0])) @ u.T @ y


def isiterable(obj: Any) -> bool:
    try:
        _ = iter(obj)
        return True
    except TypeError:
        return False
