import pytest
import numpy as np
from sampler import Sampler
from numpy.testing import assert_allclose as close

def test_init():
    def foo(x, y):
        return x+y
    sampler = Sampler(foo)
    assert len(sampler.domain) == 2
    close(sampler.domain[0], (0.0, 1.0))
    close(sampler.domain[1], (0.0, 1.0))

    (x, y), z = sampler.sample(100)
    close(x+y, z)
    assert max(x) < 1.0
    assert min(x) > 0.0
    assert min(y) > 0.0
    assert max(x) < 1.0

    a = np.arange(0, 100).reshape((10, 10))

    sampler = Sampler(a)
    assert len(sampler.domain) == 2
    close(sampler.domain[0], (0, 10))
    close(sampler.domain[1], (0, 10))

    (x, y), z = sampler.sample(100)
    print(x)
    print(y)
    print(z)
    assert np.max(x) <= 1.0
    assert np.min(x) >= 0.0
    assert np.min(y) >= 0.0
    assert np.max(x) <= 1.0
