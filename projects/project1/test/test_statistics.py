import numpy as np
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
from numpy.testing import assert_allclose
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from fysstatistics import vandermonde, Regressor
from fysstatistics import Ridge as fysRidge
from resources import franke


@pytest.fixture
def trivial():
    x = np.linspace(0, 2, 100)
    y = 5.1 + 3.2*x + 0.3*x**2
    return x, y, [5.1, 3.2, 0.3]


@pytest.fixture
def franke_noisy():
    x = np.linspace(0, 1, 50)
    y = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x, y)
    Z = franke(X, Y)
    Z_noise = Z + np.random.normal(0, 0.1, Z.shape)
    return X, Y, Z_noise


def test_vandermonde_single():
    x = np.arange(0, 10, 1)
    X = vandermonde([x], [[1, 2, 4, 10]])
    assert X.shape == (x.size, 5)
    assert_allclose(X[:, 0], np.ones_like(x))
    assert_allclose(X[:, 1], x)
    assert_allclose(X[:, 2], x**2)
    assert_allclose(X[:, 3], x**4)
    assert_allclose(X[:, 4], x**10)

def test_vandermonde_multiple():
    x = np.arange(0, 10, 1)
    y = np.linspace(-10, 22, 10)
    X = vandermonde([x, y, x], [[1, 2, 4, 10], [1, 3], [6]])
    assert X.shape == (x.size, 8)
    assert_allclose(X[:, 0], np.ones_like(x))
    assert_allclose(X[:, 1], x)
    assert_allclose(X[:, 2], x**2)
    assert_allclose(X[:, 3], x**4)
    assert_allclose(X[:, 4], x**10)
    assert_allclose(X[:, 5], y)
    assert_allclose(X[:, 6], y**3)
    assert_allclose(X[:, 7], x**6)


def test_vandemonde_exception():
    x = np.arange(0, 10, 1)
    y = np.linspace(-10, 22, 10)
    z = np.arange(0, 20, 1)
    with pytest.raises(AssertionError):
        vandermonde([x, z], [[1], [2]])
    with pytest.raises(ValueError):
        vandermonde([x, y, y], [[1], [2]])
    with pytest.raises(ValueError):
        vandermonde([x, y], [[1], [2], [3, 4]])
    with pytest.raises(ValueError):
        vandermonde([x], [])
    with pytest.raises(ValueError):
        vandermonde([], [])


def test_regressor(trivial):
    # Test fitting
    x, y, beta = trivial
    reg = Regressor(x, y)
    beta_hat = reg.fit([2])
    # Fails since the design matrix is standardized
    #assert_allclose(beta, beta_hat)

    # Test prediction
    y_hat = reg.predict(x)
    assert_allclose(y, y_hat)

    # Test training scoring
    r2 = r2_score(y, y_hat)
    assert_allclose(r2, reg.r2())
    mse = mean_squared_error(y, y_hat)
    assert_allclose(mse, reg.mse())

    # Test test scoring
    x = np.linspace(-20, 3, 64)
    y = beta[0] + beta[1]*x + beta[2]*x**2
    y_hat = reg.predict(x)
    assert_allclose(y, y_hat)  # Since we have no noise
    r2 = r2_score(y, y_hat)
    assert_allclose(r2, reg.r2(x, y))
    mse = mean_squared_error(y, y_hat)
    assert_allclose(mse, reg.mse(x, y))

    with pytest.raises(AssertionError):
        reg.mse(x)
    with pytest.raises(AssertionError):
        reg.mse(response=x)
    with pytest.raises(AssertionError):
        reg.r2(x)
    with pytest.raises(AssertionError):
        reg.r2(response=x)


def test_regressor_2d(franke_noisy):
    X, Y, Z = franke_noisy
    reg = Regressor([X, Y], Z)
    beta = reg.fit([5, 5])
    assert beta.size == 11
    Z_hat = reg.predict([X, Y])

    # SKlearn for comparison
    V = reg.design_matrix
    clf = LinearRegression(fit_intercept=False).fit(V, Z.flatten())
    assert_allclose(clf.coef_, beta)
    Z_tilde = clf.predict(V).reshape(Z_hat.shape)
    assert_allclose(Z_hat, Z_tilde)

    r2 = r2_score(Z.flatten(), Z_tilde.flatten())
    mse = mean_squared_error(Z.flatten(), Z_tilde.flatten())
    assert_allclose(r2, reg.r2())
    assert_allclose(mse, reg.mse())

    # Test confidence interval
    df = pd.DataFrame(data=reg.design_matrix, columns=['constant', 'x1', 'x2', 'x3', 'x4', 'x5',
                                                        'y1', 'y2', 'y3', 'y4', 'y5'])
    df['response'] = reg.response
    res = smf.ols("response ~ x1 + x2 + x3 + x4 + x5 + y1 + y2 + y3 + y4 + y5", data=df).fit()

    tscore = np.asarray(res.tvalues)
    assert_allclose(tscore, reg.tscore)

    ci = np.asarray(res.conf_int())
    assert_allclose(ci, reg.ci(0.95))


@pytest.mark.parametrize("alpha", [0.1, 0.3, 0.5, 1.0, 1.2, 2.2, 2.5, 3.5, 4, 5, 6, 10])
def test_ridge(franke_noisy, alpha):
    X, Y, Z = franke_noisy
    reg = fysRidge([X, Y], Z, parameter=alpha)
    beta = reg.fit([5, 5])
    Z_hat = reg.predict([X, Y])

    # SKlearn for comparison
    V = reg.design_matrix
    clf = Ridge(alpha=alpha, fit_intercept=False).fit(V, Z.flatten())
    intercept = beta[0]
    assert_allclose(clf.coef_, beta[1:])
    Z_tilde = clf.predict(V).reshape(Z_hat.shape)
    assert_allclose(Z_hat - intercept, Z_tilde)


def test_interactions():
    # Fails with matrix invesion
    x = np.asarray([1, 2, 3, 4, 5])
    y = np.asarray([3, 3, 5, 6, 7])
    z = np.asarray([3, 6, 8, 10, 12])
    reg = Regressor([x, y], z)
    beta = reg.fit([3, 2], interactions=True)
    v = reg.design_matrix
    i = 1
    def test_col(x):
        nonlocal i
        x = (x-np.mean(x))/np.std(x)
        assert_allclose(v[:, i], x)
        i += 1
    test_col(x)
    test_col(x**2)
    test_col(x**3)
    test_col(y)
    test_col(y**2)
    test_col(x*y)
    test_col(x*y**2)
    test_col(x**2*y)
    test_col(x**2*y**2)
    test_col(x**3*y)
    test_col(x**3*y**2)

    clf = LinearRegression(fit_intercept=False).fit(v, z)
    # This fails if X and Y are illformed
    assert_allclose(clf.coef_, beta)


def test_restricted_interactions():
    x = np.asarray([1, 2, 3, 4, 5])
    y = np.asarray([2, 4, 5, 6, 7])
    z = np.asarray([2, 4, 8, 10, 12])
    reg = Regressor([x, y], z)
    beta = reg.fit([3, 2], max_interaction=2)
    v = reg.design_matrix
    i = 1
    def test_col(x):
        nonlocal i
        x = (x-np.mean(x))/np.std(x)
        assert_allclose(v[:, i], x)
        i += 1
    test_col(x)
    test_col(x**2)
    test_col(x**3)
    test_col(y)
    test_col(y**2)
    test_col(x*y)

    clf = LinearRegression(fit_intercept=False).fit(v, z)
    # This fails if X and Y are illformed
    assert_allclose(clf.coef_, beta)

    reg = Regressor([x, y], z)
    beta = reg.fit([3, 2], max_interaction=5)
    v = reg.design_matrix
    i = 1
    def test_col(x):
        nonlocal i
        x = (x-np.mean(x))/np.std(x)
        assert_allclose(v[:, i], x)
        i += 1
    test_col(x)
    test_col(x**2)
    test_col(x**3)
    test_col(y)
    test_col(y**2)
    test_col(x*y)
    test_col(x*y**2)
    test_col(x**2*y)
    test_col(x**2*y**2)
    test_col(x**3*y)

    clf = LinearRegression(fit_intercept=False).fit(v, z)
    # This fails if X and Y are illformed
    assert_allclose(clf.coef_, beta)


def test_beta_dict():
    x = np.asarray([1, 2, 3, 4, 5])
    y = np.asarray([3, 3, 5, 6, 7])
    z = np.asarray([3, 6, 8, 10, 12])
    reg = Regressor([x, y], z)
    beta = reg.fit([3, 2], interactions=True)
    coeffs = reg.betadict()
    assert_allclose(beta[0], coeffs['const'])
    assert_allclose(beta[1], coeffs['x^1'])
    assert_allclose(beta[2], coeffs['x^2'])
    assert_allclose(beta[3], coeffs['x^3'])
    assert_allclose(beta[4], coeffs['y^1'])
    assert_allclose(beta[5], coeffs['y^2'])
    assert_allclose(beta[6], coeffs['x^1y^1'])
    assert_allclose(beta[7], coeffs['x^1y^2'])
    assert_allclose(beta[8], coeffs['x^2y^1'])
    assert_allclose(beta[9], coeffs['x^2y^2'])
    assert_allclose(beta[10], coeffs['x^3y^1'])


def test_design_matrix_normalization(trivial):
    x, y, beta = trivial
    reg = Regressor([x], y)
    reg.fit([3], interactions=False)
    V = reg.design_matrix

    # 1 along first column
    assert np.sum(V[:, 0], axis=0) == V.shape[0]
    # Sums to 1 along the other columns
    for col in range(1, V.shape[1]):
        assert pytest.approx(np.sum(V[:, col])) == 0

