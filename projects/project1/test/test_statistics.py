import sys
sys.path.insert(0, "../src/")
import numpy as np
import pytest
from numpy.testing import assert_allclose
from statistics import vandermonde, Regressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from resources import franke


@pytest.fixture
def trivial():
    x = np.linspace(0, 1, 100)
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
    assert_allclose(beta, beta_hat)

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
    V = reg.vandermonde
    clf = LinearRegression(fit_intercept=False).fit(V, Z.flatten())
    assert_allclose(clf.coef_, beta)
    Z_tilde = clf.predict(V).reshape(Z_hat.shape)
    assert_allclose(Z_hat, Z_tilde)

    r2 = r2_score(Z.flatten(), Z_tilde.flatten())
    mse = mean_squared_error(Z.flatten(), Z_tilde.flatten())
    assert_allclose(r2, reg.r2())
    assert_allclose(mse, reg.mse())

