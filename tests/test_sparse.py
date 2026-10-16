
import numpy as np

from tinygradlab.engine.sparse import sparse_matmul


def test_sparse_matmul_matches_dense():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(5, 4))
    weights = rng.normal(size=(4, 3))
    weights[np.abs(weights) < 0.8] = 0.0

    actual = sparse_matmul(x, weights)
    expected = x @ weights

    assert np.allclose(actual, expected)


def test_sparse_matmul_all_zero_weights():
    x = np.ones((2, 3))
    weights = np.zeros((3, 4))

    actual = sparse_matmul(x, weights)

    assert np.allclose(actual, np.zeros((2, 4)))
