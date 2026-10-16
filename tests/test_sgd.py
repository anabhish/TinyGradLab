
import numpy as np

from tinygradlab.engine.tensor import Tensor
from tinygradlab.optim.sgd import SGD


def test_sgd_updates_parameter():
    weight = Tensor([2.0, 4.0], requires_grad=True)
    weight.grad = np.array([1.0, 2.0])

    optimizer = SGD([weight], lr=0.1)
    optimizer.step()

    assert np.allclose(weight.data, [1.9, 3.8])


def test_sgd_zero_grad():
    weight = Tensor([2.0], requires_grad=True)
    weight.grad = np.array([1.0])

    optimizer = SGD([weight])
    optimizer.zero_grad()

    assert weight.grad is None




def test_sgd_momentum_update():
    from tinygradlab.engine.tensor import Tensor
    from tinygradlab.optim.sgd import SGD
    import numpy as np

    weight = Tensor([1.0], requires_grad=True)
    optimizer = SGD([weight], lr=0.1, momentum=0.9)

    weight.grad = np.array([2.0])
    optimizer.step()
    assert np.allclose(weight.data, [0.8])

    optimizer.zero_grad()
    weight.grad = np.array([1.0])
    optimizer.step()

    # velocity = 0.9 * 2 + 1 = 2.8
    # weight = 0.8 - 0.1 * 2.8 = 0.52
    assert np.allclose(weight.data, [0.52])


def test_sgd_momentum_stays_zero_for_pruned_weights():
    from tinygradlab.engine.tensor import Tensor
    from tinygradlab.optim.sgd import SGD
    import numpy as np

    weight = Tensor([1.0, 2.0], requires_grad=True)
    mask = np.array([1.0, 0.0])
    masks = {weight: mask}

    optimizer = SGD([weight], lr=0.1, momentum=0.9, masks=masks)

    weight.grad = np.array([1.0, 100.0])
    optimizer.step()

    assert np.allclose(weight.data, [0.9, 0.0])
    assert np.allclose(optimizer.velocity[weight], [1.0, 0.0])
    assert np.allclose(weight.grad, [1.0, 0.0])



def test_sgd_resets_momentum_when_weight_is_revived():
    import numpy as np
    from tinygradlab.engine.tensor import Tensor
    from tinygradlab.optim.sgd import SGD

    weight = Tensor([1.0], requires_grad=True)
    masks = {weight: np.array([1.0])}
    optimizer = SGD(
        [weight], lr=0.1, momentum=0.9, masks=masks
    )

    weight.grad = np.array([2.0])
    optimizer.step()
    assert np.allclose(optimizer.velocity[weight], [2.0])

    masks[weight] = np.array([0.0])
    weight.grad = np.array([5.0])
    optimizer.step()
    assert np.allclose(weight.data, [0.0])
    assert np.allclose(optimizer.velocity[weight], [0.0])

    masks[weight] = np.array([1.0])
    weight.grad = np.array([1.0])
    optimizer.step()

    assert np.allclose(optimizer.velocity[weight], [1.0])
    assert np.allclose(weight.data, [-0.1])


def test_masked_weight_forward_backward_prune_train_revive_end_to_end():
    """A masked edge has zero effective weight/gradient and can learn after revival."""
    weight = Tensor([1.0], requires_grad=True)
    mask = np.array([1.0])
    masks = {weight: mask.copy()}
    optimizer = SGD([weight], lr=0.1, momentum=0.9, masks=masks)
    input_value = Tensor([2.0])

    # Establish nonzero momentum while the connection is active.
    effective_weight = weight * Tensor(masks[weight])
    loss = (input_value * effective_weight).sum()
    loss.backward()
    np.testing.assert_allclose(weight.grad, [2.0])
    optimizer.step()
    np.testing.assert_allclose(optimizer.velocity[weight], [2.0])

    # Prune the connection; the real graph must produce zero output and gradient.
    masks[weight] = np.array([0.0])
    effective_weight = weight * Tensor(masks[weight])
    loss = (input_value * effective_weight).sum()
    np.testing.assert_allclose(loss.data, 0.0)
    loss.backward()
    np.testing.assert_allclose(weight.grad, [0.0])
    optimizer.step()
    np.testing.assert_allclose(weight.data, [0.0])
    np.testing.assert_allclose(optimizer.velocity[weight], [0.0])

    # Additional steps while pruned must not accumulate optimizer state.
    for _ in range(3):
        optimizer.zero_grad()
        effective_weight = weight * Tensor(masks[weight])
        loss = (input_value * effective_weight).sum()
        loss.backward()
        optimizer.step()
        np.testing.assert_allclose(weight.data, [0.0])
        np.testing.assert_allclose(optimizer.velocity[weight], [0.0])

    # Revive it: fresh gradient updates it without stale pre-pruning momentum.
    masks[weight] = np.array([1.0])
    optimizer.zero_grad()
    effective_weight = weight * Tensor(masks[weight])
    loss = (input_value * effective_weight).sum()
    loss.backward()
    np.testing.assert_allclose(weight.grad, [2.0])
    optimizer.step()
    np.testing.assert_allclose(optimizer.velocity[weight], [2.0])
    np.testing.assert_allclose(weight.data, [-0.2])
