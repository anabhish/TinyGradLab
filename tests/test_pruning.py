
import numpy as np
import pytest

from tinygradlab.engine.tensor import Tensor
from tinygradlab.nn.pruning import magnitude_prune
from tinygradlab.optim.sgd import SGD


def test_magnitude_pruning_removes_smallest_weights():
    weights = Tensor([1.0, -0.1, 0.5], requires_grad=True)
    other = Tensor([2.0, -0.2], requires_grad=True)

    masks = magnitude_prune([weights, other], sparsity=0.4)

    assert np.count_nonzero(weights.data == 0.0) == 1
    assert np.count_nonzero(other.data == 0.0) == 1
    assert np.count_nonzero(masks[weights] == 0.0) == 1
    assert np.count_nonzero(masks[other] == 0.0) == 1


def test_sgd_keeps_pruned_weights_zero():
    weights = Tensor([1.0, 2.0], requires_grad=True)
    masks = {weights: np.array([1.0, 0.0])}

    weights.data *= masks[weights]
    weights.grad = np.array([1.0, 100.0])

    optimizer = SGD([weights], lr=0.1, masks=masks)
    optimizer.step()

    assert np.allclose(weights.data, [0.9, 0.0])
    assert np.allclose(weights.grad, [1.0, 0.0])


def test_magnitude_pruning_rejects_invalid_sparsity():
    weights = Tensor([1.0, 2.0], requires_grad=True)

    with pytest.raises(ValueError):
        magnitude_prune([weights], sparsity=1.0)


from tinygradlab.nn.pruning import calculate_sparsity


def test_calculate_sparsity():
    weights = Tensor([1.0, 0.0, 0.0, 4.0], requires_grad=True)

    sparsity = calculate_sparsity([weights])

    assert np.isclose(sparsity, 0.5)


from tinygradlab.nn.pruning import update_masks_by_magnitude


def test_progressive_pruning_preserves_old_masks():
    weights = Tensor([0.1, 0.2, 0.3, 0.4], requires_grad=True)
    masks = {}

    update_masks_by_magnitude([weights], masks, 0.25)
    first_mask = masks[weights].copy()

    update_masks_by_magnitude([weights], masks, 0.50)

    assert np.count_nonzero(masks[weights] == 0.0) == 2
    assert np.all(masks[weights][first_mask == 0.0] == 0.0)
    assert np.all(weights.data[masks[weights] == 0.0] == 0.0)


def test_progressive_pruning_rejects_lower_target():
    weights = Tensor([0.1, 0.2, 0.3, 0.4], requires_grad=True)
    masks = {}

    update_masks_by_magnitude([weights], masks, 0.50)

    with pytest.raises(ValueError):
        update_masks_by_magnitude([weights], masks, 0.25)
