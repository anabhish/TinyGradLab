"""Measure dense versus sparse-aware inference cost honestly.

The sparse path skips zero weights using sparse_matmul. The dense baseline
still uses NumPy matrix multiplication over the same masked dense array.
"""
import csv
import time
from pathlib import Path

import numpy as np

from tinygradlab.engine.sparse import sparse_matmul


ROOT = Path(__file__).resolve().parents[1]


def dense_forward(x, weights):
    return x @ weights


def sparse_forward(x, weights):
    return sparse_matmul(x, weights)


def benchmark(function, x, weights, repeats=20, rounds=5):
    function(x, weights)  # warm-up
    timings = []
    for _ in range(rounds):
        start = time.perf_counter()
        for _ in range(repeats):
            function(x, weights)
        timings.append((time.perf_counter() - start) / repeats)
    return float(np.median(timings))


def make_pruned_weights(weights, sparsity):
    pruned = weights.copy()
    count = int(sparsity * pruned.size)
    if count:
        indices = np.argsort(np.abs(pruned).ravel(), kind="stable")[:count]
        pruned.ravel()[indices] = 0.0
    return pruned


def main():
    rng = np.random.default_rng(42)
    batch_size, in_features, out_features = 32, 128, 128
    sparsities = [0.0, 0.50, 0.75, 0.90, 0.95]
    x = rng.normal(size=(batch_size, in_features))
    original_weights = rng.normal(size=(in_features, out_features))
    rows = []

    print(
        f"{'Target':>7} {'Actual':>7} {'Active':>7} "
        f"{'Dense MACs':>12} {'Sparse MACs':>12} "
        f"{'Dense ms':>10} {'Sparse ms':>11} {'Dense/Sparse':>13}"
    )

    for target in sparsities:
        weights = make_pruned_weights(original_weights, target)
        total = weights.size
        active = int(np.count_nonzero(weights))
        actual = 1.0 - active / total
        dense_result = dense_forward(x, weights)
        sparse_result = sparse_forward(x, weights)
        np.testing.assert_allclose(dense_result, sparse_result, rtol=1e-10, atol=1e-10)

        dense_time = benchmark(dense_forward, x, weights)
        sparse_time = benchmark(sparse_forward, x, weights)
        dense_macs = batch_size * in_features * out_features
        sparse_macs = batch_size * active
        ratio = dense_time / sparse_time

        row = {
            "target_sparsity": target,
            "achieved_sparsity": actual,
            "total_weights": total,
            "active_weights": active,
            "dense_macs_per_batch": dense_macs,
            "sparse_macs_per_batch": sparse_macs,
            "dense_median_ms": dense_time * 1000,
            "sparse_aware_median_ms": sparse_time * 1000,
            "dense_over_sparse_speed_ratio": ratio,
            "numerically_equivalent": True,
            "batch_size": batch_size,
            "input_features": in_features,
            "output_features": out_features,
            "timing_repeats_per_round": 20,
            "timing_rounds": 5,
        }
        rows.append(row)
        print(
            f"{target:>6.0%} {actual:>6.1%} {active:>7d} "
            f"{dense_macs:>12d} {sparse_macs:>12d} "
            f"{dense_time*1000:>9.3f} {sparse_time*1000:>10.3f} "
            f"{ratio:>12.5f}x"
        )

    path = ROOT / "pruning_cost_benchmark.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {path.name}")
    print(
        "Interpretation: sparse MAC counts fall with active weights, but "
        "Python-loop sparse execution may be slower than dense NumPy. "
        "Do not claim wall-clock speedup unless the measured ratio exceeds 1."
    )

    # Measure the actual pruned XOR model exported by train_mlp.py.
    model_path = ROOT / "pruned_xor_model.npz"
    if not model_path.exists():
        raise FileNotFoundError(
            "Run `python examples/train_mlp.py` first to create "
            "pruned_xor_model.npz."
        )

    with np.load(model_path) as artifact:
        w1 = artifact["layer1_weight"].copy()
        b1 = artifact["layer1_bias"].copy()
        w2 = artifact["layer2_weight"].copy()
        b2 = artifact["layer2_bias"].copy()
        mask1 = artifact["layer1_mask"].copy()
        mask2 = artifact["layer2_mask"].copy()

    xor_x = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])

    def dense_mlp_forward(batch):
        hidden = np.maximum(batch @ w1 + b1, 0.0)
        return hidden @ w2 + b2

    def sparse_mlp_forward(batch):
        hidden = np.maximum(sparse_matmul(batch, w1) + b1, 0.0)
        return sparse_matmul(hidden, w2) + b2

    dense_logits = dense_mlp_forward(xor_x)
    sparse_logits = sparse_mlp_forward(xor_x)
    np.testing.assert_allclose(dense_logits, sparse_logits, rtol=1e-10, atol=1e-10)

    total_weights = w1.size + w2.size
    active_weights = int(np.count_nonzero(mask1) + np.count_nonzero(mask2))
    actual_model_sparsity = 1.0 - active_weights / total_weights
    dense_model_macs = len(xor_x) * (w1.size + w2.size)
    sparse_model_macs = len(xor_x) * active_weights
    dense_model_time = benchmark(
        lambda batch, unused: dense_mlp_forward(batch), xor_x, w1
    )
    sparse_model_time = benchmark(
        lambda batch, unused: sparse_mlp_forward(batch), xor_x, w1
    )
    model_cost_row = {
        "model": "trained_pruned_xor_mlp",
        "achieved_weight_sparsity": actual_model_sparsity,
        "total_weights": total_weights,
        "active_weights": active_weights,
        "dense_macs_per_full_dataset": dense_model_macs,
        "sparse_macs_per_full_dataset": sparse_model_macs,
        "dense_median_ms": dense_model_time * 1000,
        "sparse_aware_median_ms": sparse_model_time * 1000,
        "dense_over_sparse_speed_ratio": dense_model_time / sparse_model_time,
        "dense_sparse_outputs_equivalent": True,
        "measurement_examples": len(xor_x),
        "timing_repeats_per_round": 20,
        "timing_rounds": 5,
    }
    model_cost_path = ROOT / "pruned_xor_model_cost.csv"
    with model_cost_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(model_cost_row.keys()))
        writer.writeheader()
        writer.writerow(model_cost_row)

    print("\nActual trained pruned XOR MLP:")
    print(
        f"achieved sparsity={actual_model_sparsity:.2%}, "
        f"active weights={active_weights}/{total_weights}, "
        f"dense MACs={dense_model_macs}, sparse-aware MACs={sparse_model_macs}"
    )
    print(
        f"dense median={dense_model_time * 1000:.4f} ms, "
        f"sparse-aware median={sparse_model_time * 1000:.4f} ms, "
        f"dense/sparse ratio={dense_model_time / sparse_model_time:.5f}x"
    )
    print(f"Saved {model_cost_path.name}")


if __name__ == "__main__":
    main()
