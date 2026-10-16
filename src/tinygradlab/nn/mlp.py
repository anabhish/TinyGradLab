
from tinygradlab.nn.linear import Linear


class MLP:
    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        seed=42,
    ):
        self.layer1 = Linear(in_features, hidden_features, seed=seed)
        self.layer2 = Linear(
            hidden_features,
            out_features,
            seed=seed + 1,
        )

    def __call__(self, x):
        hidden = self.layer1(x).relu()
        return self.layer2(hidden)

    def parameters(self):
        return self.layer1.parameters() + self.layer2.parameters()
