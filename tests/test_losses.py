
import numpy as np

from tinygradlab.engine.tensor import Tensor
from tinygradlab.nn.losses import binary_cross_entropy_with_logits
from tinygradlab.nn.losses import softmax_cross_entropy


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

def test_softmax_cross_entropy_is_finite_for_extreme_logits():
    logits = Tensor(
        [[1000.0, 1001.0, 999.0],
         [-1000.0, -999.0, -1001.0]],
        requires_grad=True,
    )

    loss = softmax_cross_entropy(logits, np.array([1, 1]))

    assert np.isfinite(loss.data).all()
    assert loss.data.item() >= 0.0


def test_softmax_cross_entropy_backward():
    logits = Tensor([[2.0, 1.0, 0.1]], requires_grad=True)

    loss = softmax_cross_entropy(logits, np.array([0]))
    loss.backward()

    assert logits.grad.shape == logits.data.shape
    assert np.isfinite(logits.grad).all()
    assert np.allclose(np.sum(logits.grad, axis=1), [0.0])


def test_softmax_cross_entropy_gradient_matches_finite_difference():
    values = np.array([[0.2, -0.3, 0.7]])
    targets = np.array([2])
    logits = Tensor(values.copy(), requires_grad=True)

    softmax_cross_entropy(logits, targets).backward()

    epsilon = 1e-5
    plus = values.copy()
    minus = values.copy()
    plus[0, 0] += epsilon
    minus[0, 0] -= epsilon

    numerical = (
        softmax_cross_entropy(Tensor(plus), targets).data.item()
        - softmax_cross_entropy(Tensor(minus), targets).data.item()
    ) / (2.0 * epsilon)

    assert np.allclose(logits.grad[0, 0], numerical, atol=1e-6)

def test_binary_cross_entropy_gradient_matches_finite_difference():
    values = np.array([-0.4, 0.8])
    targets = np.array([0.0, 1.0])
    logits = Tensor(values.copy(), requires_grad=True)
    binary_cross_entropy_with_logits(logits, targets).backward()

    epsilon = 1e-5
    plus = values.copy()
    minus = values.copy()
    plus[1] += epsilon
    minus[1] -= epsilon
    numerical = (
        binary_cross_entropy_with_logits(Tensor(plus), targets).data.item()
        - binary_cross_entropy_with_logits(Tensor(minus), targets).data.item()
    ) / (2 * epsilon)
    np.testing.assert_allclose(logits.grad[1], numerical, atol=1e-6)
