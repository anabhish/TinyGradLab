
import numpy as np

from tinygradlab.engine.tensor import Tensor
from tinygradlab.nn.mlp import MLP


def test_mlp_forward_shape():
    model = MLP(2, 8, 1)
    x = Tensor(np.ones((4, 2)))

    output = model(x)

    assert output.data.shape == (4, 1)


def test_mlp_parameters_receive_gradients():
    model = MLP(2, 8, 1)
    x = Tensor(np.ones((4, 2)))

    loss = model(x).sum()
    loss.backward()

    for param in model.parameters():
        assert param.grad is not None
        assert param.grad.shape == param.data.shape
