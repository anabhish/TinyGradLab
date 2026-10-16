
import numpy as np


def sparse_matmul(x, weights):
    """Compute x @ weights by iterating over nonzero weights.

    x:       (batch_size, in_features)
    weights: (in_features, out_features)
    """
    x = np.asarray(x, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    if x.ndim != 2 or weights.ndim != 2:
        raise ValueError("Both inputs must be 2D arrays.")

    if x.shape[1] != weights.shape[0]:
        raise ValueError("Matrix dimensions do not match.")

    output = np.zeros((x.shape[0], weights.shape[1]))

    rows, cols = np.nonzero(weights)
    values = weights[rows, cols]

    for row, col, value in zip(rows, cols, values):
        output[:, col] += x[:, row] * value

    return output
