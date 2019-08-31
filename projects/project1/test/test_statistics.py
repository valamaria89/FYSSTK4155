import sys
sys.path.insert(0, "../src/")
import numpy as np
import pytest
from numpy.testing import assert_allclose
from statistics import vandermonde


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

