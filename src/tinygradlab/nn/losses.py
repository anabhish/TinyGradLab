
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



def softmax_cross_entropy(logits, targets):
    """Compute numerically stable softmax cross-entropy.

    Args:
        logits: Tensor of shape (N, C), where N is batch size
                and C is number of classes.
        targets: Integer class labels of shape (N,).

    Returns:
        Scalar Tensor containing mean cross-entropy loss.
    """
    if not isinstance(logits, Tensor):
        raise TypeError("logits must be a Tensor")

    if logits.data.ndim != 2:
        raise ValueError("logits must be a 2D tensor of shape (N, C)")

    targets = np.asarray(targets)

    batch_size, num_classes = logits.data.shape

    if targets.shape != (batch_size,):
        raise ValueError("targets must have shape (N,)")

    if not np.issubdtype(targets.dtype, np.integer):
        if not np.all(targets == np.floor(targets)):
            raise ValueError("targets must contain integer class labels")
        targets = targets.astype(np.int64)

    targets = targets.astype(np.int64)

    if np.any(targets < 0) or np.any(targets >= num_classes):
        raise ValueError("target class index is out of range")

    # Subtract each row's maximum for numerical stability.
    shifted = logits.data - np.max(logits.data, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    probabilities = exp_logits / np.sum(
        exp_logits, axis=1, keepdims=True
    )

    log_normalizer = np.log(
        np.sum(exp_logits, axis=1)
    )
    per_example_loss = (
        log_normalizer - shifted[np.arange(batch_size), targets]
    )
    loss_value = np.mean(per_example_loss)

    result = Tensor(
        loss_value,
        requires_grad=logits.requires_grad,
        _parents=(logits,),
        _op="softmax_cross_entropy",
    )

    def backward():
        if logits.requires_grad:
            grad = probabilities.copy()
            grad[np.arange(batch_size), targets] -= 1.0
            grad *= result.grad / batch_size

            if logits.grad is None:
                logits.grad = np.zeros_like(logits.data)

            logits.grad += grad

    result._backward = backward
    return result