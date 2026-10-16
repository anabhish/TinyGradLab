
class SGD:
    def __init__(self, parameters, lr=0.01, momentum=0.0, masks=None):
        if lr <= 0:
            raise ValueError("Learning rate must be positive.")
        if not 0 <= momentum < 1:
            raise ValueError("Momentum must be in [0, 1).")

        self.parameters = list(parameters)
        self.lr = lr
        self.momentum = momentum
        self.masks = masks if masks is not None else {}

        # Keep one velocity array for each parameter.
        self.velocity = {
            param: __import__("numpy").zeros_like(param.data)
            for param in self.parameters
        }

        #remember the previous masks
        self.previous_masks = {
            param: mask.copy()
            for param, mask in self.masks.items()
        }

    def step(self):
        for param in self.parameters:
            mask = self.masks.get(param)
            velocity = self.velocity[param]

            if mask is not None:
                previous_mask = self.previous_masks.get(param)

                if previous_mask is not None:
                    # Reset momentum when a pruned connection is revived.
                    revived = (previous_mask == 0) & (mask != 0)
                    velocity[revived] = 0.0

                # Keep currently pruned weights and momentum at zero.
                param.data *= mask
                velocity *= mask

            if param.grad is not None:
                grad = param.grad

                if mask is not None:
                    grad *= mask

                velocity *= self.momentum
                velocity += grad

                param.data -= self.lr * velocity

                if mask is not None:
                    velocity *= mask
                    param.data *= mask

            # Remember the mask for the next optimizer step.
            if mask is not None:
                self.previous_masks[param] = mask.copy()

    def zero_grad(self):
        for param in self.parameters:
            param.grad = None
