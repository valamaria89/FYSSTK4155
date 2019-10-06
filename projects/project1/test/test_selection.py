import pytest
import numpy as np
from sklearn.model_selection import KFold
from numpy.testing import assert_allclose
from modelselection import kfold, kfold_indices


@pytest.fixture
def trivial():
    x = np.linspace(1, 10, 30)
    y = np.linspace(2, 3, 20)
    X, Y = np.meshgrid(x, y)
    Z = X+Y
    kf = KFold(n_splits=4)
    return X, Y, Z, kf


def test_indices(trivial):
    X, _, _, kf = trivial
    for ref, cand in zip(kf.split(X), kfold_indices(X, k=4)):
        rtrain, rtest = ref
        ctrain, ctest = cand
        assert_allclose(rtrain, ctrain)
        assert_allclose(rtest, ctest)


def test_selection(trivial):
    X, Y, Z, kf = trivial
    for ref, cand in zip(kf.split(X), kfold([X, Y], Z, k=4)):
        rtrain, rtest = ref
        (XYtrain, XYtest), (Ztrain, Ztest) = cand
        (Xtrain, Ytrain) = XYtrain
        (Xtest, Ytest) = XYtest
        assert_allclose(X[rtrain], Xtrain)
        assert_allclose(Y[rtrain], Ytrain)
        assert_allclose(Z[rtrain], Ztrain)
        assert_allclose(X[rtest], Xtest)
        assert_allclose(Y[rtest], Ytest)
        assert_allclose(Z[rtest], Ztest)


@pytest.mark.parametrize("n_splits", [n for n in range(2, 100)])
def test_various(n_splits):
    x = np.linspace(1, 10, 100)
    y = np.linspace(-5, 2.4, 100)
    X, Y = np.meshgrid(x, y)
    kf = KFold(n_splits=n_splits)
    for ref, cand in zip(kf.split(X), kfold_indices(X, k=n_splits)):
        rtrain, rtest = ref
        ctrain, ctest = cand
        assert_allclose(rtrain, ctrain)
        assert_allclose(rtest, ctest)

