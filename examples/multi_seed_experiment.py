"""Matched multi-seed pruning comparison for Parts 3 and 4."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tinygradlab.engine.tensor import Tensor
from tinygradlab.nn.losses import binary_cross_entropy_with_logits
from tinygradlab.nn.metrics import binary_accuracy
from tinygradlab.nn.mlp import MLP
from tinygradlab.nn.pruning import (
    calculate_sparsity,
    magnitude_prune,
    taylor_saliency_prune,
    update_masks_by_magnitude,
    update_masks_by_taylor_saliency,
)
from tinygradlab.optim.sgd import SGD


ROOT = Path(__file__).resolve().parents[1]
SEEDS = [11, 22, 33, 44, 55]
TARGETS = [0.0, 0.50, 0.75, 0.90, 0.95]
METHODS = [
    "one-shot-magnitude",
    "progressive-magnitude",
    "one-shot-taylor",
    "progressive-taylor",
]


def dataset():
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])
    y = np.array([[0.0], [1.0], [1.0], [0.0]])
    return X, y


def cubic_sparsity(epoch, start, end, final_sparsity):
    """Cubic ramp from zero to the requested final sparsity."""
    if epoch <= start:
        return 0.0
    if epoch >= end:
        return final_sparsity
    progress = (epoch - start) / (end - start)
    return final_sparsity * (1.0 - (1.0 - progress) ** 3)


def run_trial(seed, target, method, epochs=2000):
    X, y = dataset()
    model = MLP(2, 40, 1, seed=seed)
    weights = [model.layer1.weight, model.layer2.weight]
    masks = {param: np.ones_like(param.data) for param in weights}
    optimizer = SGD(model.parameters(), lr=0.1, masks=masks)

    if method not in METHODS:
        raise ValueError(f"Unknown pruning method: {method}")

    progressive = method.startswith("progressive")
    use_taylor = method.endswith("taylor")

    if progressive:
        prune_epochs = set(range(550, 1501, 50))
    else:
        prune_epochs = {500}

    for epoch in range(epochs):
        if epoch in prune_epochs:
            current_target = (
                cubic_sparsity(epoch, 500, 1500, target)
                if progressive else target
            )
            if use_taylor:
                optimizer.zero_grad()
                saliency_logits = model(Tensor(X))
                saliency_loss = binary_cross_entropy_with_logits(
                    saliency_logits, y
                )
                saliency_loss.backward()
                if progressive:
                    update_masks_by_taylor_saliency(
                        weights, masks, current_target
                    )
                else:
                    masks.update(taylor_saliency_prune(weights, current_target))
                    for param in weights:
                        param.data *= masks[param]
            elif progressive:
                update_masks_by_magnitude(
                    weights, masks, current_target
                )
            else:
                masks.update(magnitude_prune(weights, current_target))
                for param in weights:
                    param.data *= masks[param]

        optimizer.zero_grad()
        logits = model(Tensor(X))
        loss = binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        optimizer.step()

    final_logits = model(Tensor(X)).data
    final_loss = binary_cross_entropy_with_logits(
        Tensor(final_logits), y
    )
    actual_sparsity = calculate_sparsity(weights)

    expected_zeros = int(target * sum(param.data.size for param in weights))
    actual_zeros = sum(
        int(np.count_nonzero(masks[param] == 0.0)) for param in weights
    )
    if actual_zeros != expected_zeros:
        raise AssertionError(
            f"{method}, target={target}: expected {expected_zeros} masked "
            f"weights, got {actual_zeros}"
        )
    for param in weights:
        np.testing.assert_array_equal(
            param.data[masks[param] == 0.0],
            np.zeros_like(param.data[masks[param] == 0.0]),
        )

    return {
        "accuracy": float(binary_accuracy(final_logits, y)),
        "loss": float(final_loss.data.item()),
        "actual_sparsity": float(actual_sparsity),
        "active_parameters": int(sum(
            np.count_nonzero(masks[param]) for param in weights
        )),
        "total_parameters": int(sum(param.data.size for param in weights)),
    }


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    trial_rows = []
    for target in TARGETS:
        for method in METHODS:
            for seed in SEEDS:
                result = run_trial(seed, target, method)
                trial_rows.append({
                    "seed": seed,
                    "method": method,
                    "target_sparsity": target,
                    **result,
                })
                print(
                    f"seed={seed:2d} target={target:>4.0%} "
                    f"{method:<22} accuracy={result['accuracy']:.0%} "
                    f"loss={result['loss']:.5f} "
                    f"achieved={result['actual_sparsity']:.1%}"
                )

    summary_rows = []
    for target in TARGETS:
        for method in METHODS:
            group = [
                row for row in trial_rows
                if row["method"] == method
                and row["target_sparsity"] == target
            ]
            accuracies = np.array([row["accuracy"] for row in group])
            losses = np.array([row["loss"] for row in group])
            sparsities = np.array([row["actual_sparsity"] for row in group])
            summary_rows.append({
                "method": method,
                "target_sparsity": target,
                "n_seeds": len(group),
                "mean_accuracy": float(accuracies.mean()),
                "std_accuracy": float(accuracies.std(ddof=1)),
                "mean_loss": float(losses.mean()),
                "std_loss": float(losses.std(ddof=1)),
                "mean_achieved_sparsity": float(sparsities.mean()),
                "mean_active_parameters": float(np.mean(
                    [row["active_parameters"] for row in group]
                )),
                "total_parameters": group[0]["total_parameters"],
            })

    # Paired per-seed differences: positive values favor Taylor.
    paired_rows = []
    for target in TARGETS:
        for schedule in ("one-shot", "progressive"):
            magnitude_method = f"{schedule}-magnitude"
            taylor_method = f"{schedule}-taylor"
            mag_by_seed = {
                row["seed"]: row["accuracy"] for row in trial_rows
                if row["target_sparsity"] == target
                and row["method"] == magnitude_method
            }
            tay_by_seed = {
                row["seed"]: row["accuracy"] for row in trial_rows
                if row["target_sparsity"] == target
                and row["method"] == taylor_method
            }
            diffs = np.array([
                tay_by_seed[seed] - mag_by_seed[seed]
                for seed in SEEDS
            ])
            mean_diff = float(diffs.mean())
            sd_diff = float(diffs.std(ddof=1))
            se = sd_diff / np.sqrt(len(diffs))
            paired_rows.append({
                "target_sparsity": target,
                "comparison": f"{taylor_method}-minus-{magnitude_method}",
                "n_paired_seeds": len(diffs),
                "mean_accuracy_difference": mean_diff,
                "std_paired_difference": sd_diff,
                "approx_95ci_low": mean_diff - 1.96 * se,
                "approx_95ci_high": mean_diff + 1.96 * se,
            })

    write_csv(ROOT / "pruning_trials.csv", trial_rows)
    write_csv(ROOT / "pruning_results.csv", summary_rows)
    write_csv(ROOT / "pruning_paired_comparison.csv", paired_rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    for method in METHODS:
        group = sorted(
            [row for row in summary_rows if row["method"] == method],
            key=lambda row: row["mean_achieved_sparsity"],
        )
        ax.errorbar(
            [row["mean_achieved_sparsity"] for row in group],
            [row["mean_accuracy"] for row in group],
            yerr=[row["std_accuracy"] for row in group],
            marker="o",
            capsize=3,
            label=method,
        )
    ax.set_xlabel("Achieved weight sparsity")
    ax.set_ylabel("Mean training accuracy (± 1 SD across seeds)")
    ax.set_title("Pruning accuracy–sparsity Pareto comparison")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "pruning_pareto.png", dpi=160)
    plt.close(fig)

    print("\nSummary (mean ± sample SD across seeds):")
    for row in summary_rows:
        print(
            f"{row['method']:<22} {row['target_sparsity']:>4.0%} "
            f"accuracy={row['mean_accuracy']:.1%}±{row['std_accuracy']:.1%} "
            f"loss={row['mean_loss']:.5f} "
            f"sparsity={row['mean_achieved_sparsity']:.1%}"
        )
    print(
        "\nSaved pruning_trials.csv, pruning_results.csv, "
        "pruning_paired_comparison.csv, pruning_pareto.png"
    )


if __name__ == "__main__":
    main()
