
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tinygradlab.engine.tensor import Tensor
from tinygradlab.nn.mlp import MLP
from tinygradlab.nn.losses import binary_cross_entropy_with_logits
from tinygradlab.nn.metrics import binary_accuracy
from tinygradlab.nn.pruning import (
    update_masks_by_taylor_saliency,
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


def cubic_sparsity(epoch, start, end, final_sparsity):
    """Cubic target ramp: slow start, faster middle, smooth finish."""
    if epoch <= start:
        return 0.0
    if epoch >= end:
        return final_sparsity
    progress = (epoch - start) / (end - start)
    return final_sparsity * (1.0 - (1.0 - progress) ** 3)


def main():
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])
    y = np.array([[0.0], [1.0], [1.0], [0.0]])

    model = MLP(2, 40, 1)
    weights = [model.layer1.weight, model.layer2.weight]
    masks = {param: np.ones_like(param.data) for param in weights}

    optimizer = SGD(model.parameters(), lr=0.1, masks=masks)
    rng = np.random.default_rng(42)

    epochs = 2000
    batch_size = 2
    pruning_start = 500
    pruning_end = 1500
    pruning_interval = 50
    target_sparsity = 0.90

    loss_history = []
    accuracy_history = []
    sparsity_history = []

    check_parameters_finite(model)

    for epoch in range(epochs):
        if (
            pruning_start < epoch <= pruning_end
            and (epoch - pruning_start) % pruning_interval == 0
        ):
            target = cubic_sparsity(
                epoch, pruning_start, pruning_end, target_sparsity
            )
            # Compute gradients on the full training set for Taylor saliency.
            optimizer.zero_grad()

            pruning_logits = model(Tensor(X))
            check_finite("pruning logits", pruning_logits.data)

            pruning_loss = binary_cross_entropy_with_logits(pruning_logits, y)
            check_finite("pruning loss", pruning_loss.data)

            pruning_loss.backward()
            check_gradients_finite(model)

            update_masks_by_taylor_saliency(
                weights,
                masks,
                target,
            )

            post_prune_predictions = model(Tensor(X)).data
            post_prune_accuracy = binary_accuracy(post_prune_predictions, y)
            print(
                f"Pruning epoch {epoch:4d} | Target {target:.1%} "
                f"| Achieved {calculate_sparsity(weights):.1%} "
                f"| Accuracy {post_prune_accuracy:.1%}"
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

    figure, axes = plt.subplots(1, 3, figsize=(15, 4))

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

    axes[2].plot(epochs_axis, np.asarray(sparsity_history) * 100)
    axes[2].set_title("Weight Sparsity")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Sparsity (%)")
    axes[2].set_ylim(0, 100)
    axes[2].grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

    check_parameters_finite(model)
    expected_sparsity = target_sparsity
    actual_sparsity = calculate_sparsity(weights)

    if not np.isclose(actual_sparsity, expected_sparsity):
        raise AssertionError(
            f"Expected {expected_sparsity:.0%} weight sparsity, "
            f"got {actual_sparsity:.2%}"
        )

    for param in weights:
        pruned_values = param.data[masks[param] == 0.0]
        np.testing.assert_array_equal(
            pruned_values,
            np.zeros_like(pruned_values),
        )

    print(f"\nTraining completed: {epochs} epochs.")
    print(f"Final loss: {loss_history[-1]:.6f}")
    print(f"Final training accuracy: {accuracy_history[-1]:.2%}")
    print(f"Final sparsity: {sparsity_history[-1]:.2%}")
    print(f"Learning curve saved to: {output_path}")

    model_path = Path(__file__).resolve().parents[1] / "pruned_xor_model.npz"
    np.savez_compressed(
        model_path,
        layer1_weight=model.layer1.weight.data,
        layer1_bias=model.layer1.bias.data,
        layer2_weight=model.layer2.weight.data,
        layer2_bias=model.layer2.bias.data,
        layer1_mask=masks[model.layer1.weight],
        layer2_mask=masks[model.layer2.weight],
    )
    print(f"Pruned model weights and masks saved to: {model_path}")


if __name__ == "__main__":
    main()
