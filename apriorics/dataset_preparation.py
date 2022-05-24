from math import ceil
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
from nptyping import NDArray


def split_data_k_fold(
    data: Sequence[Any],
    k: int = 5,
    test_size: Union[int, float] = 0.1,
    seed: Optional[int] = None,
) -> Dict[str, NDArray[Any]]:
    r"""
    Split input data sequence into k folds and a test fold.

    Args:
        data: input data sequence.
        k: number of folds.
        test_size: if int, number of items to keep for test; if float, portion of data
            to keep for test.
        seed: seed to use for RNG.

    Returns:
        A dictionary that maps fold numbers (or "test") to the corresponding data
        sequence.
    """
    rng = np.random.default_rng(seed)
    n = len(data)
    data = np.array(data)
    if isinstance(test_size, float):
        test_size = ceil(test_size * n)
    test_idxs = rng.randint(n, size=test_size)
    test_data = data[test_idxs]

    data = np.delete(data, test_idxs)
    folds = rng.randint(k, size=len(data))

    splits = {str(i): data[folds == i] for i in range(k)}
    splits["test"] = test_data

    return splits
