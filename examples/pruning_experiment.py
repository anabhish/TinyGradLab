
import numpy as np

from tinygradlab.engine.tensor import Tensor
from tinygradlab.nn.mlp import MLP
from tinygradlab.nn.losses import binary_cross_entropy_with_logits
from tinygradlab.nn.metrics import binary_accuracy
from tinygradlab.nn.pruning import magnitude_prune, calculate_sparsity
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
    optimizer = SGD(model.parameters(), lr=0.1)

    # Train the baseline model.
    for _ in range(2000):
        optimizer.zero_grad()
        logits = model(Tensor(X))
        loss = binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        optimizer.step()

    def evaluate():
        logits = model(Tensor(X)).data
        return binary_accuracy(logits, y)

    # Only prune weight matrices, not biases.
    weights = [model.layer1.weight, model.layer2.weight]

    print(f"Baseline accuracy: {evaluate():.2%}")
    print(f"{'Target':>10} {'Actual':>10} {'Accuracy':>12} {'Nonzero':>10}")

    for target_sparsity in [0.0, 0.25, 0.50, 0.75]:
        # Reinitialize a fresh copy of the trained model for each
        # target so each point measures pruning from the same baseline.
        candidate = MLP(2, 8, 1)
        candidate.layer1.weight.data[:] = model.layer1.weight.data
        candidate.layer1.bias.data[:] = model.layer1.bias.data
        candidate.layer2.weight.data[:] = model.layer2.weight.data
        candidate.layer2.bias.data[:] = model.layer2.bias.data

        candidate_weights = [
            candidate.layer1.weight,
            candidate.layer2.weight,
        ]

        if target_sparsity > 0:
            magnitude_prune(candidate_weights, target_sparsity)

        actual = calculate_sparsity(candidate_weights)
        accuracy = binary_accuracy(candidate(Tensor(X)).data, y)
        nonzero = sum(np.count_nonzero(p.data) for p in candidate_weights)

        print(
            f"{target_sparsity:>9.0%} "
            f"{actual:>9.0%} "
            f"{accuracy:>11.2%} "
            f"{nonzero:>10d}"
        )


if __name__ == "__main__":
    main()
