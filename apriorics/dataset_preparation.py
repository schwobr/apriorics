import numpy as np
from math import ceil


def split_data_k_fold(data, k=5, test_size=0.1, seed=None):
    rng = np.random.default_rng(seed)
    n = len(data)
    data = np.array(data)
    if isinstance(test_size, float):
        test_size = ceil(test_size * n)
    test_idxs = rng.randint(n, size=test_size)
    test_data = data[test_idxs]

    data = np.delete(data, test_idxs)
    folds = rng.randint(k, size=len(data))

    splits = {i: data[folds == i] for i in range(k)}
    splits["test"] = test_data

    return splits
