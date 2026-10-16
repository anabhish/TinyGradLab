
from pathlib import Path

import matplotlib.pyplot as plt
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


def check_finite(name, value):
    """Fail immediately if a value contains NaN or infinity."""
    if not np.all(np.isfinite(value)):
        raise FloatingPointError(
            f"Non-finite values detected in {name}."
        )


def check_parameters_finite(model):
    for index, parameter in enumerate(model.parameters()):
        check_finite(f"parameter {index}", parameter.data)


def check_gradients_finite(model):
    for index, parameter in enumerate(model.parameters()):
        if parameter.grad is not None:
            check_finite(f"gradient {index}", parameter.grad)


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
    rng = np.random.default_rng(42)

    epochs = 2000
    batch_size = 2
    pruning_schedule = {
        500: 0.25,
        1000: 0.50,
        1500: 0.75,
    }

    loss_history = []
    accuracy_history = []
    sparsity_history = []

    check_parameters_finite(model)

    for epoch in range(epochs):
        if epoch in pruning_schedule:
            update_masks_by_magnitude(
                weights,
                masks,
                pruning_schedule[epoch],
            )

        indices = rng.permutation(len(X))
        epoch_loss = 0.0

        for start in range(0, len(X), batch_size):
            batch_indices = indices[start:start + batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            optimizer.zero_grad()

            logits = model(Tensor(X_batch))
            check_finite("training logits", logits.data)

            loss = binary_cross_entropy_with_logits(logits, y_batch)
            check_finite("training loss", loss.data)

            loss.backward()
            check_gradients_finite(model)

            optimizer.step()
            check_parameters_finite(model)

            epoch_loss += float(loss.data.item()) * len(batch_indices)

        epoch_loss /= len(X)

        # Evaluate the complete training set after this epoch.
        predictions = model(Tensor(X)).data
        check_finite("evaluation predictions", predictions)

        accuracy = binary_accuracy(predictions, y)
        check_finite("training accuracy", np.asarray(accuracy))

        sparsity = calculate_sparsity(weights)

        loss_history.append(epoch_loss)
        accuracy_history.append(float(accuracy))
        sparsity_history.append(float(sparsity))

        if epoch % 200 == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch:4d} | Loss {epoch_loss:.4f} "
                f"| Accuracy {accuracy:.2%} "
                f"| Sparsity {sparsity:.2%}"
            )

    # Save reproducible learning-curve evidence in the project root.
    output_path = (
        Path(__file__).resolve().parents[1] / "training_curve.png"
    )

    epochs_axis = np.arange(1, epochs + 1)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs_axis, loss_history)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary cross-entropy")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_axis, np.asarray(accuracy_history) * 100)
    axes[1].set_title("Training Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim(0, 105)
    axes[1].grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

    check_parameters_finite(model)

    print(f"\nTraining completed: {epochs} epochs.")
    print(f"Final loss: {loss_history[-1]:.6f}")
    print(f"Final training accuracy: {accuracy_history[-1]:.2%}")
    print(f"Final sparsity: {sparsity_history[-1]:.2%}")
    print(f"Learning curve saved to: {output_path}")


if __name__ == "__main__":
    main()
