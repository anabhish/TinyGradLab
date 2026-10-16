
import time
import numpy as np
from tinygradlab.engine.sparse import sparse_matmul


def dense_forward(x, weights):
    return x @ weights


# def sparse_forward(x, weights):
#     # Keep only columns containing at least one nonzero weight.
#     active_columns = np.any(weights != 0, axis=0)
#
#     if not np.any(active_columns):
#         return np.zeros((x.shape[0], weights.shape[1]))
#
#     active_weights = weights[:, active_columns]
#     active_output = x @ active_weights
#
#     output = np.zeros((x.shape[0], weights.shape[1]))
#     output[:, active_columns] = active_output
#     return output

def sparse_forward(x, weights):
    return sparse_matmul(x, weights)



def benchmark(function, x, weights, repeats=30, rounds=5):
    # Warm up the function before measuring.
    function(x, weights)

    timings = []

    for _ in range(rounds):
        start = time.perf_counter()

        for _ in range(repeats):
            function(x, weights)

        elapsed = time.perf_counter() - start
        timings.append(elapsed / repeats)

    # Median reduces the influence of unusually slow timing rounds.
    return float(np.median(timings))


def main():
    rng = np.random.default_rng(42)

    # batch_size = 256
    # in_features = 512
    # out_features = 512
    # sparsities = [0.0, 0.25, 0.50, 0.75]

    batch_size = 32
    in_features = 128
    out_features = 128
    sparsities = [0.0, 0.50, 0.75]

    x = rng.normal(size=(batch_size, in_features))
    original_weights = rng.normal(
        size=(in_features, out_features)
    )

    print(
        f"{'Target':>8} {'Actual':>8} {'Dense (ms)':>12} "
        f"{'Sparse (ms)':>13} {'Speedup':>12}"
    )

    for sparsity in sparsities:
        weights = make_pruned_weights(original_weights, sparsity)
        actual_sparsity = np.mean(weights == 0.0)

        dense_result = dense_forward(x, weights)
        sparse_result = sparse_forward(x, weights)

        assert np.allclose(dense_result, sparse_result)

        dense_time = benchmark(dense_forward, x, weights)
        sparse_time = benchmark(sparse_forward, x, weights)

        speedup = dense_time / sparse_time

        print(
            f"{sparsity:>7.0%} "
            f"{actual_sparsity:>7.2%} "
            f"{dense_time * 1000:>11.3f} "
            f"{sparse_time * 1000:>12.3f} "
            f"{speedup:>11.5f}x"
        )

def make_pruned_weights(weights, sparsity):
    pruned = weights.copy()
    count = int(sparsity * pruned.size)

    if count:
        indices = np.argpartition(
            np.abs(pruned).ravel(), count - 1
        )[:count]
        pruned.ravel()[indices] = 0.0

    return pruned



if __name__ == "__main__":
    main()
