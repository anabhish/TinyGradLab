
import numpy as np

from tinygradlab.engine.tensor import Tensor


class Linear:
    def __init__(self, in_features, out_features, seed=42):
        rng = np.random.default_rng(seed)

        # He-style scale for initializing weights
        scale = np.sqrt(2.0 / in_features)

        self.weight = Tensor(
            rng.normal(0.0, scale, size=(in_features, out_features)),
            requires_grad=True,
        )

        self.bias = Tensor(
            np.zeros(out_features),
            requires_grad=True,
        )

    def __call__(self, x):
        return x @ self.weight + self.bias

    def parameters(self):
        return [self.weight, self.bias]

