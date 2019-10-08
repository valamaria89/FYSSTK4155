import numpy as np
import inspect
from numpy import ndarray
from typing import Union, Callable, Optional, Iterable, Tuple, List


class Sampler:
    """ Samples from a function or array

    Samples uniformly or uniformly randomly from a function
    or a array in order to expose a uniform call syntax
    """
    def __init__(self, source: Union[Callable[..., float], ndarray],
                 domain: Optional[Iterable[Tuple[float, float]]] = None) -> None:
        if isinstance(source, ndarray):
            if domain is None:
                domain = [(0, length) for length in source.shape]
            def _source(*args):
                # Convert from (0, 1) -> (start, stop)
                X = []
                for x, (start, stop) in zip(args, domain):
                    X.append(((stop - start)*x + start).astype(int))
                X = tuple(X)
                return source[X]
            self.source = _source
            self._source = source
        else:
            if domain is None:
                # Set the domain to [0, 1] for each positional parameter
                domain = []
                args = inspect.signature(source)
                for param in args.parameters.values():
                    if ((param.kind == inspect.Parameter.POSITIONAL_ONLY) or
                        (param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD)):
                        domain.append((0.0, 1.0))
                self.source = source
        self.domain = domain
        self.do_add_noise = False
        self.sigma = 0.0
        self.type = type(source)

    def __call__(self, num_samples: int,
                 random=True) -> Tuple[List[ndarray], ndarray]:
        return self.sample(num_samples, random)

    def sample(self, num_samples: int,
               random=True) -> Tuple[List[ndarray], ndarray]:
        # Construct the parameter range
        if self.type == ndarray:
            X = self.sample_array(num_samples, random)
        else:
            X = self.sample_func(num_samples, random)

        y = self.source(*X)
        return X, y + self.add_noise(y)

    def sample_array(self, num_samples: int, random: bool):
        X = []
        indices = []
        for start, stop in self.domain:
            if random:
                _x = np.arange(start, stop, 1, dtype=int)
                x = np.random.choice(_x, num_samples, replace=True)
            else:
                x = np.linspace(start, stop, num_samples,
                                dtype=int)
            # Normalize the sample to prevent numerical errors
            indices.append(x)
            x = (x - start)/(stop - start)
            X.append(x)

        return X

    def sample_func(self, num_samples: int, random: bool):
        X = []
        for start, stop in self.domain:
            if random:
                x = start + stop*np.random.rand(num_samples)
            else:
                x = np.linspace(start, stop, num_samples)
            X.append(x)
        return X

    def add_noise(self, y: ndarray) -> ndarray:
        if self.do_add_noise:
            return np.random.normal(0, self.sigma, y.shape)
        else:
            return np.zeros_like(y)

    def set_noise(self, sigma: float = 0.1) -> None:
        if sigma > 0:
            self.do_add_noise = True
        else:
            self.do_add_noise = False
        self.sigma = sigma

    def population_sample(self) -> ndarray:
        """ Gives an as large as possible sample"""
        if self.type == ndarray:
            x = np.arange(*self.domain[0])
            y = np.arange(*self.domain[1])
            X, Y = np.meshgrid(x, y)
            Z = self._source[X, Y]

            x = (x - np.min(x))/(np.max(x) - np.min(x))
            y = (y - np.min(y))/(np.max(y) - np.min(y))
            X, Y = np.meshgrid(x, y)
            return [X, Y], Z
        else:
            x = np.linspace(0, 1, 1000)
            y = np.linspace(0, 1, 999)
            X, Y = np.meshgrid(x, y)
            return (X, Y), self.source(X, Y)
