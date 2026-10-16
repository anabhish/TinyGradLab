
import numpy as np


def magnitude_prune(parameters, sparsity):
    """Prune the smallest-magnitude weights globally."""
    if not 0.0 <= sparsity < 1.0:
        raise ValueError("sparsity must be in [0, 1)")

    parameters = list(parameters)

    if not parameters:
        raise ValueError("parameters must not be empty")

    total = sum(param.data.size for param in parameters)
    prune_count = int(sparsity * total)

    magnitudes = np.concatenate([
        np.abs(param.data).ravel()
        for param in parameters
    ])

    prune_indices = np.argsort(magnitudes, kind="stable")[:prune_count]

    masks = {}
    offset = 0

    for param in parameters:
        size = param.data.size
        mask = np.ones_like(param.data)
        local_indices = prune_indices[
            (prune_indices >= offset)
            & (prune_indices < offset + size)
        ] - offset

        mask.ravel()[local_indices] = 0.0
        param.data *= mask

        masks[param] = mask
        offset += size

    return masks

def calculate_sparsity(parameters):
    """Return the fraction of parameter elements that are exactly zero."""
    parameters = list(parameters)

    total = sum(param.data.size for param in parameters)

    if total == 0:
        raise ValueError("parameters must contain at least one element")

    zero_count = sum(
        np.count_nonzero(param.data == 0.0)
        for param in parameters
    )

    return zero_count / total


def update_masks_by_magnitude(parameters, masks, target_sparsity):
    """Increase global weight sparsity while preserving existing masks."""
    parameters = list(parameters)

    if not parameters:
        raise ValueError("parameters must not be empty")

    if not 0.0 <= target_sparsity < 1.0:
        raise ValueError("target_sparsity must be in [0, 1)")

    total = sum(param.data.size for param in parameters)

    for param in parameters:
        if param not in masks:
            masks[param] = np.ones_like(param.data)
        elif masks[param].shape != param.data.shape:
            raise ValueError("mask shape must match parameter shape")

        param.data *= masks[param]

    current_zeros = sum(
        np.count_nonzero(mask == 0.0)
        for mask in masks.values()
        if any(mask is existing for existing in masks.values())
    )

    # Count only the parameters being pruned in this operation.
    current_zeros = sum(
        np.count_nonzero(masks[param] == 0.0)
        for param in parameters
    )

    target_zeros = int(target_sparsity * total)
    additional = target_zeros - current_zeros

    if additional < 0:
        raise ValueError("target sparsity cannot be lower than current sparsity")

    if additional == 0:
        return masks

    active = []
    for param in parameters:
        active_indices = np.flatnonzero(masks[param].ravel() != 0.0)
        active_values = np.abs(param.data.ravel()[active_indices])
        active.extend(
            (value, param, int(index))
            for value, index in zip(active_values, active_indices)
        )

    if additional > len(active):
        raise ValueError("not enough active weights to reach target sparsity")

    active.sort(key=lambda item: item[0])

    for _, param, index in active[:additional]:
        masks[param].ravel()[index] = 0.0

    for param in parameters:
        param.data *= masks[param]

    return masks

def taylor_saliency_prune(parameters, sparsity):
    """Prune weights with the lowest first-order Taylor saliency."""
    if not 0.0 <= sparsity < 1.0:
        raise ValueError("sparsity must be in [0, 1)")

    parameters = list(parameters)

    if not parameters:
        raise ValueError("parameters must not be empty")

    if any(param.grad is None for param in parameters):
        raise ValueError("all parameters must have gradients")

    total = sum(param.data.size for param in parameters)
    prune_count = int(sparsity * total)

    scores = np.concatenate([
        np.abs(param.data * param.grad).ravel()
        for param in parameters
    ])

    prune_indices = np.argsort(scores, kind="stable")[:prune_count]

    masks = {}
    offset = 0

    for param in parameters:
        size = param.data.size
        mask = np.ones_like(param.data)

        local_indices = prune_indices[
            (prune_indices >= offset)
            & (prune_indices < offset + size)
        ] - offset

        mask.ravel()[local_indices] = 0.0
        param.data *= mask

        masks[param] = mask
        offset += size

    return masks

def update_masks_by_taylor_saliency(parameters, masks, target_sparsity):
    """Increase sparsity using Taylor saliency while preserving old masks."""
    parameters = list(parameters)

    if not parameters:
        raise ValueError("parameters must not be empty")

    if not 0.0 <= target_sparsity < 1.0:
        raise ValueError("target_sparsity must be in [0, 1)")

    if any(param.grad is None for param in parameters):
        raise ValueError("all parameters must have gradients")

    total = sum(param.data.size for param in parameters)

    for param in parameters:
        if param not in masks:
            masks[param] = np.ones_like(param.data)
        elif masks[param].shape != param.data.shape:
            raise ValueError("mask shape must match parameter shape")

        param.data *= masks[param]

    current_zeros = sum(
        np.count_nonzero(masks[param] == 0.0)
        for param in parameters
    )

    target_zeros = int(target_sparsity * total)
    additional = target_zeros - current_zeros

    if additional < 0:
        raise ValueError("target sparsity cannot be lower than current sparsity")

    if additional == 0:
        return masks

    active = []
    for param in parameters:
        active_indices = np.flatnonzero(masks[param].ravel() != 0.0)
        scores = np.abs(param.data * param.grad).ravel()[active_indices]
        active.extend(
            (score, param, int(index))
            for score, index in zip(scores, active_indices)
        )

    if additional > len(active):
        raise ValueError("not enough active weights to reach target sparsity")

    active.sort(key=lambda item: item[0])

    for _, param, index in active[:additional]:
        masks[param].ravel()[index] = 0.0

    for param in parameters:
        param.data *= masks[param]

    return masks