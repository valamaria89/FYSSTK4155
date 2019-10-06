import numpy as np
from numpy import ndarray
from typing import Sequence, Iterator, Tuple, List


def kfold(predictors: Sequence[ndarray], response: ndarray, k: int = 1,
          shuffle: bool = False) -> Iterator[List[Tuple[ndarray, ndarray]]]:
    """ Perform K-fold cross validation on N predictors

    Args:
        predictors: Equal length predictors to select from
        response: The response
        k: The number of partitions
        shuffle: Whether to randomly shuffle the entries
    Yields:
        Each iteration yields a list over each predictor where
        each element is a tuple on the form (training sample, test sample)
        for the i'th sample.
    """
    # Ensure the arrays are of the same size
    predictors = [np.asarray(array) for array in predictors]
    for array in predictors:
        if array.shape[0] != predictors[0].shape[0]:
            raise ValueError("Shape missmatch in predictors")
    if predictors[0].shape[0] != response.shape[0]:
        raise ValueError("Shape missmatch between predictor and response")

    if shuffle:
        indices = np.arange(0, predictors[0].shape[0])
        np.random.shuffle(indices)
        shuffled = [predictor[indices] for predictor in predictors]
        predictors = shuffled
        response = response[indices]

    for train, test in kfold_indices(predictors[0], k=k):
        pred_train = [pred[train] for pred in predictors]
        pred_test = [pred[test] for pred in predictors]
        r = (response[train], response[test])
        yield (pred_train, pred_test), r


def kfold_indices(predictor: ndarray,
                  k: int = 1) -> Iterator[Tuple[ndarray, ndarray]]:
    """ Create the indices for k-fold cross validation

    Args:
        predictor: A prototype predictor to split
        k: The number of partitions
    Yields:
        Each iteration yields a tuple on the form
            (training sample, test sample)
        for the i'th sample.
    """
    # First n_samples % n_splits have length n_samples // n_split + 1
    length = predictor.shape[0]
    indices = np.arange(0, length)
    n_larger_splits = length % k
    larger_splits = length // k + 1
    for i in range(n_larger_splits):
        test = np.asarray(range(larger_splits*i, larger_splits*(i+1)))
        mask = np.ones_like(indices, dtype=bool)
        mask[test] = False
        train = indices[mask]
        yield train, test

    # The rest have length n_samples // n_split
    split = length // k
    offset = n_larger_splits
    for i in range(n_larger_splits, k):
        test = np.asarray(range(split*i + offset, split*(i+1) + offset))
        mask = np.ones_like(indices, dtype=bool)
        mask[test] = False
        train = indices[mask]
        yield train, test
