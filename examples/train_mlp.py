
import numpy as np

from tinygradlab.engine.tensor import Tensor
from tinygradlab.nn.mlp import MLP
from tinygradlab.nn.losses import binary_cross_entropy_with_logits
from tinygradlab.nn.metrics import binary_accuracy
from tinygradlab.nn.pruning import (
    update_masks_by_magnitude,
    calculate_sparsity,
)
from tinygradlab.optim.sgd import SGD


def main():
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])
    y = np.array([[0.0], [1.0], [1.0], [0.0]])

    model = MLP(2, 8, 1)
    weights = [model.layer1.weight, model.layer2.weight]
    masks = {param: np.ones_like(param.data) for param in weights}

    optimizer = SGD(model.parameters(), lr=0.1, masks=masks)

    epochs = 2000
    pruning_schedule = {
        500: 0.25,
        1000: 0.50,
        1500: 0.75,
    }

    for epoch in range(epochs):
        if epoch in pruning_schedule:
            update_masks_by_magnitude(
                weights,
                masks,
                pruning_schedule[epoch],
            )

        optimizer.zero_grad()
        logits = model(Tensor(X))
        loss = binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        optimizer.step()

        if epoch % 200 == 0:
            accuracy = binary_accuracy(model(Tensor(X)).data, y)
            sparsity = calculate_sparsity(weights)
            print(
                f"Epoch {epoch:4d} | Loss {loss.data.item():.4f} "
                f"| Accuracy {accuracy:.2%} | Sparsity {sparsity:.2%}"
            )

    print("\nFinal accuracy:", f"{binary_accuracy(model(Tensor(X)).data, y):.2%}")
    print("Final weight sparsity:", f"{calculate_sparsity(weights):.2%}")


if __name__ == "__main__":
    main()
