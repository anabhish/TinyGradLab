
import numpy as np

from tinygradlab.engine.tensor import Tensor
from tinygradlab.nn.losses import binary_cross_entropy_with_logits


def test_binary_cross_entropy_is_finite_for_extreme_logits():
    logits = Tensor(
        np.array([-1000.0, 1000.0]),
        requires_grad=True,
    )
    targets = np.array([0.0, 1.0])

    loss = binary_cross_entropy_with_logits(logits, targets)

    assert np.isfinite(loss.data).all()
    assert np.allclose(loss.data, 0.0, atol=1e-12)


def test_binary_cross_entropy_backward():
    logits = Tensor(np.array([0.0, 0.0]), requires_grad=True)
    targets = np.array([0.0, 1.0])

    loss = binary_cross_entropy_with_logits(logits, targets)
    loss.backward()

    assert np.allclose(logits.grad, [0.25, -0.25])
