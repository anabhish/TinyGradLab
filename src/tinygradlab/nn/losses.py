
import numpy as np

from tinygradlab.engine.tensor import Tensor


def binary_cross_entropy_with_logits(logits, targets):
    if not isinstance(logits, Tensor):
        raise TypeError("logits must be a Tensor")

    targets = np.asarray(targets, dtype=np.float64)

    if logits.data.shape != targets.shape:
        raise ValueError("logits and targets must have the same shape")

    if not np.all((targets == 0.0) | (targets == 1.0)):
        raise ValueError("targets must contain only 0 or 1")

    z = logits.data
    per_element = (
        np.maximum(z, 0.0)
        - z * targets
        + np.log1p(np.exp(-np.abs(z)))
    )
    loss_value = np.mean(per_element)

    result = Tensor(
        loss_value,
        requires_grad=logits.requires_grad,
        _parents=(logits,),
        _op="binary_cross_entropy_with_logits",
    )

    def backward():
        if logits.requires_grad:
            # Stable sigmoid, avoiding overflow for negative logits.
            probs = np.empty_like(z)
            positive = z >= 0.0
            probs[positive] = 1.0 / (1.0 + np.exp(-z[positive]))

            exp_z = np.exp(z[~positive])
            probs[~positive] = exp_z / (1.0 + exp_z)

            grad = result.grad * (probs - targets) / targets.size

            if logits.grad is None:
                logits.grad = np.zeros_like(logits.data)

            logits.grad += grad

    result._backward = backward
    return result
