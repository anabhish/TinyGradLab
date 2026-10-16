
import numpy as np
import pytest

from tinygradlab.nn.metrics import binary_accuracy


def test_binary_accuracy():
    logits = np.array([[-2.0], [3.0], [1.0], [-1.0]])
    targets = np.array([[0.0], [1.0], [1.0], [1.0]])

    assert binary_accuracy(logits, targets) == 0.75


def test_binary_accuracy_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        binary_accuracy(np.array([1.0, -1.0]), np.array([[1.0], [0.0]]))
