
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
