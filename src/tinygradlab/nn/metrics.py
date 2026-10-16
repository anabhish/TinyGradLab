
import numpy as np


def binary_accuracy(logits, targets):
    logits_data = np.asarray(
        logits.data if hasattr(logits, "data") else logits
    )
    targets = np.asarray(targets)

    if logits_data.shape != targets.shape:
        raise ValueError("Logits and targets must have matching shapes.")

    if not np.all((targets == 0) | (targets == 1)):
        raise ValueError("Targets must contain only 0 and 1.")

    predictions = logits_data >= 0.0
    return float(np.mean(predictions == targets))
