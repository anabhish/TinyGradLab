
import numpy as np

from tinygradlab.engine.tensor import Tensor
from tinygradlab.nn.linear import Linear


def test_linear_forward_shape():
    layer = Linear(in_features=3, out_features=2)
    x = Tensor(np.ones((4, 3)))

    result = layer(x)

    assert result.data.shape == (4, 2)


def test_linear_parameters_track_gradients():
    layer = Linear(in_features=3, out_features=2)
    x = Tensor(np.ones((4, 3)))

    loss = layer(x).sum()
    loss.backward()

    assert layer.weight.grad is not None
    assert layer.bias.grad is not None
    assert layer.weight.grad.shape == (3, 2)
    assert layer.bias.grad.shape == (2,)

def test_linear_parameters():
    layer = Linear(in_features=3, out_features=2)

    params = layer.parameters()

    assert len(params) == 2
    assert params[0] is layer.weight
    assert params[1] is layer.bias
    assert all(param.requires_grad for param in params)
