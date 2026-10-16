
import csv
import numpy as np
import matplotlib.pyplot as plt


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


def dataset():
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])
    y = np.array([[0.0], [1.0], [1.0], [0.0]])
    return X, y


def run_trial(seed, target, method, epochs=2000):
    X, y = dataset()
    model = MLP(2, 8, 1, seed=seed)
    weights = [model.layer1.weight, model.layer2.weight]
    masks = {p: np.ones_like(p.data) for p in weights}
    optimizer = SGD(model.parameters(), lr=0.1, masks=masks)

    if method == "one-shot":
        schedule = {500: target}
    else:
        schedule = {
            500: target / 3,
            1000: 2 * target / 3,
            1500: target,
        }

    for epoch in range(epochs):
        if epoch in schedule:
            if method == "one-shot":
                new_masks = magnitude_prune(weights, target)
                masks.update(new_masks)
            else:
                update_masks_by_magnitude(weights, masks, schedule[epoch])

        optimizer.zero_grad()
        logits = model(Tensor(X))
        loss = binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        optimizer.step()

    final_logits = model(Tensor(X)).data

    return {
        "accuracy": binary_accuracy(final_logits, y),
        "loss": float(loss.data.item()),
        "sparsity": calculate_sparsity(weights),
    }


def main():
    seeds = [11, 22, 33, 44, 55]
    targets = [0.25, 0.50, 0.75]
    methods = ["one-shot", "progressive"]

    trial_rows = []
    summary_rows = []

    for target in targets:
        for method in methods:
            for seed in seeds:
                result = run_trial(seed, target, method)

                trial_rows.append({
                    "seed": seed,
                    "method": method,
                    "target_sparsity": target,
                    "accuracy": result["accuracy"],
                    "loss": result["loss"],
                    "actual_sparsity": result["sparsity"],
                })

            # Summarize this method and target across all seeds.
            group = [
                row for row in trial_rows
                if row["method"] == method
                and row["target_sparsity"] == target
            ]

            accuracies = np.array([row["accuracy"] for row in group])
            losses = np.array([row["loss"] for row in group])
            sparsities = np.array(
                [row["actual_sparsity"] for row in group]
            )

            summary_rows.append({
                "method": method,
                "target_sparsity": target,
                "mean_accuracy": float(accuracies.mean()),
                "std_accuracy": float(accuracies.std()),
                "mean_loss": float(losses.mean()),
                "mean_sparsity": float(sparsities.mean()),
            })

    print(
        f"{'Method':<12} {'Target':>8} {'Mean acc':>10} "
        f"{'Std acc':>9} {'Mean loss':>11} {'Mean sparse':>12}"
    )

    for row in summary_rows:
        print(
            f"{row['method']:<12} "
            f"{row['target_sparsity']:>7.0%} "
            f"{row['mean_accuracy']:>9.2%} "
            f"{row['std_accuracy']:>8.2%} "
            f"{row['mean_loss']:>11.4f} "
            f"{row['mean_sparsity']:>11.2%}"
        )

    # Save all 30 individual trials.
    with open("pruning_trials.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=trial_rows[0].keys())
        writer.writeheader()
        writer.writerows(trial_rows)

    # Save the six aggregate rows.
    with open("pruning_results.csv", "w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=summary_rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    # Plot mean accuracy against actual mean sparsity.
    plt.figure()

    for method in methods:
        group = sorted(
            [row for row in summary_rows if row["method"] == method],
            key=lambda row: row["mean_sparsity"],
        )

        plt.plot(
            [row["mean_sparsity"] for row in group],
            [row["mean_accuracy"] for row in group],
            marker="o",
            label=method,
        )

    plt.xlabel("Weight sparsity")
    plt.ylabel("Mean training accuracy")
    plt.title("Pruning accuracy vs. sparsity")
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("pruning_pareto.png", dpi=150)
    plt.close()

    print(
        "\nSaved pruning_trials.csv, pruning_results.csv, "
        "and pruning_pareto.png"
    )


if __name__ == "__main__":
    main()

