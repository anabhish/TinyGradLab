
import numpy as np

from tinygradlab.engine.tensor import Tensor
from tinygradlab.nn.losses import binary_cross_entropy_with_logits
from tinygradlab.nn.metrics import binary_accuracy
from tinygradlab.nn.mlp import MLP
from tinygradlab.optim.sgd import SGD


def run_trial(seed, momentum):
    X = Tensor([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])

    y = np.array([[0.0], [1.0], [1.0], [0.0]])

    model = MLP(2, 8, 1, seed=seed)
    optimizer = SGD(
        model.parameters(),
        lr=0.1,
        momentum=momentum,
    )

    for _ in range(2000):
        logits = model(X)
        loss = binary_cross_entropy_with_logits(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_logits = model(X)
    final_loss = binary_cross_entropy_with_logits(final_logits, y)

    return (
        float(binary_accuracy(final_logits, y)),
        float(final_loss.data),
    )


def main():
    seeds = [11, 22, 33, 44, 55]
    momentum_values = [0.0, 0.9]

    print(
        f"{'Momentum':>10} {'Mean accuracy':>15} "
        f"{'Mean loss':>12}"
    )

    for momentum in momentum_values:
        results = [
            run_trial(seed, momentum)
            for seed in seeds
        ]

        accuracies = np.array([r[0] for r in results])
        losses = np.array([r[1] for r in results])

        print(
            f"{momentum:>10.1f} "
            f"{accuracies.mean():>14.2%} "
            f"{losses.mean():>12.4f}"
        )


if __name__ == "__main__":
    main()
