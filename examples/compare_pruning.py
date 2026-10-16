
import numpy as np

from tinygradlab.engine.tensor import Tensor
from tinygradlab.nn.mlp import MLP
from tinygradlab.nn.losses import binary_cross_entropy_with_logits
from tinygradlab.nn.metrics import binary_accuracy
from tinygradlab.nn.pruning import (
    magnitude_prune,
    update_masks_by_magnitude,
    calculate_sparsity,
)
from tinygradlab.optim.sgd import SGD


def make_dataset():
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])
    y = np.array([[0.0], [1.0], [1.0], [0.0]])
    return X, y


def train_strategy(strategy, initial_model, X, y, epochs=2000):
    model = MLP(2, 8, 1)

    # Copy the same initial weights for a fair comparison.
    for target, source in zip(model.parameters(), initial_model.parameters()):
        target.data[:] = source.data

    weights = [model.layer1.weight, model.layer2.weight]
    masks = {param: np.ones_like(param.data) for param in weights}
    optimizer = SGD(model.parameters(), lr=0.1, masks=masks)

    schedule = {500: 0.25, 1000: 0.50, 1500: 0.75}

    for epoch in range(epochs):
        if strategy == "one-shot" and epoch == 500:
            masks.update(magnitude_prune(weights, 0.75))

        elif strategy == "progressive" and epoch in schedule:
            update_masks_by_magnitude(weights, masks, schedule[epoch])

        optimizer.zero_grad()
        logits = model(Tensor(X))
        loss = binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        optimizer.step()

    final_logits = model(Tensor(X)).data

    return {
        "strategy": strategy,
        "accuracy": binary_accuracy(final_logits, y),
        "loss": loss.data.item(),
        "sparsity": calculate_sparsity(weights),
        "nonzero": sum(np.count_nonzero(p.data) for p in weights),
    }


def main():
    X, y = make_dataset()
    initial_model = MLP(2, 8, 1)

    results = [
        train_strategy("one-shot", initial_model, X, y),
        train_strategy("progressive", initial_model, X, y),
    ]

    print(
        f"{'Strategy':<14} {'Accuracy':>10} "
        f"{'Loss':>10} {'Sparsity':>10} {'Nonzero':>10}"
    )

    for result in results:
        print(
            f"{result['strategy']:<14} "
            f"{result['accuracy']:>9.2%} "
            f"{result['loss']:>10.4f} "
            f"{result['sparsity']:>9.2%} "
            f"{result['nonzero']:>10d}"
        )


if __name__ == "__main__":
    main()
