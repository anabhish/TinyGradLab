
import numpy as np


def sum_to_shape(grad, shape):
    """Sum broadcasted gradient dimensions to the original shape."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad


class Tensor:
    def __init__(
        self,
        data,
        requires_grad=False,
        _parents=(),
        _op="",
    ):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = None

        # Tensors that created this tensor
        self._parents = set(_parents)

        # Operation that created this tensor
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    def backward(self):
        if self.data.size != 1:
            raise ValueError(
                "backward() currently requires a scalar output"
            )

        topo = []
        visited = set()

        def build_topo(tensor):
            if tensor not in visited:
                visited.add(tensor)

                for parent in tensor._parents:
                    build_topo(parent)

                topo.append(tensor)

        build_topo(self)

        # Clear gradients throughout this graph for this backward pass.
        for tensor in topo:
            tensor.grad = None

        # Seed the scalar output gradient.
        self.grad = np.ones_like(self.data)

        for tensor in reversed(topo):
            tensor._backward()

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        result = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _parents=(self, other),
            _op="add",
        )

        def backward():
            if self.requires_grad:
                grad = sum_to_shape(result.grad, self.data.shape)
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += grad

            if other.requires_grad:
                grad = sum_to_shape(result.grad, other.data.shape)
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                other.grad += grad

        result._backward = backward
        return result



    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        result = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _parents=(self, other),
            _op="mul",
        )

        def backward():
            if self.requires_grad:
                grad = sum_to_shape(
                    result.grad * other.data,
                    self.data.shape,
                )
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += grad

            if other.requires_grad:
                grad = sum_to_shape(
                    result.grad * self.data,
                    other.data.shape,
                )
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                other.grad += grad

        result._backward = backward
        return result

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        result = Tensor(
            self.data - other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _parents=(self, other),
            _op="sub",
        )

        def backward():
            if self.requires_grad:
                grad = sum_to_shape(result.grad, self.data.shape)
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += grad

            if other.requires_grad:
                grad = sum_to_shape(-result.grad, other.data.shape)
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                other.grad += grad

        result._backward = backward
        return result

    def sum(self, axis=None, keepdims=False):
        result = Tensor(
            np.sum(self.data, axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _parents=(self,),
            _op="sum",
        )

        def backward():
            if self.requires_grad:
                grad = result.grad

                if axis is not None and not keepdims:
                    axes = (axis,) if isinstance(axis, int) else tuple(axis)
                    normalized_axes = tuple(
                        a if a >= 0 else a + self.data.ndim
                        for a in axes
                    )

                    for a in sorted(normalized_axes):
                        grad = np.expand_dims(grad, axis=a)

                grad = np.broadcast_to(grad, self.data.shape)

                if self.grad is None:
                    self.grad = np.zeros_like(self.data)

                self.grad += grad

        result._backward = backward
        return result

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        result = Tensor(
            self.data / other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _parents=(self, other),
            _op="div",
        )

        def backward():
            if self.requires_grad:
                grad = sum_to_shape(
                    result.grad / other.data,
                    self.data.shape,
                )
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += grad

            if other.requires_grad:
                grad = sum_to_shape(
                    -result.grad * self.data / (other.data ** 2),
                    other.data.shape,
                )
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                other.grad += grad

        result._backward = backward
        return result

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        if self.data.ndim != 2 or other.data.ndim != 2:
            raise ValueError("Matrix multiplication currently requires 2D tensors")

        result = Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _parents=(self, other),
            _op="matmul",
        )

        def backward():
            if self.requires_grad:
                grad = result.grad @ other.data.T
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += grad

            if other.requires_grad:
                grad = self.data.T @ result.grad
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                other.grad += grad

        result._backward = backward
        return result

    def relu(self):
        result = Tensor(
            np.maximum(0, self.data),
            requires_grad=self.requires_grad,
            _parents=(self,),
            _op="relu",
        )

        def backward():
            if self.requires_grad:
                grad = result.grad * (self.data > 0)
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += grad

        result._backward = backward
        return result

    def tanh(self):
        output = np.tanh(self.data)

        result = Tensor(
            output,
            requires_grad=self.requires_grad,
            _parents=(self,),
            _op="tanh",
        )

        def backward():
            if self.requires_grad:
                grad = result.grad * (1.0 - output ** 2)

                if self.grad is None:
                    self.grad = np.zeros_like(self.data)

                self.grad += grad

        result._backward = backward
        return result

    def sigmoid(self):
        # Numerically stable sigmoid calculation
        output = np.empty_like(self.data)
        positive = self.data >= 0

        output[positive] = 1.0 / (
                1.0 + np.exp(-self.data[positive])
        )

        exp_x = np.exp(self.data[~positive])
        output[~positive] = exp_x / (1.0 + exp_x)

        result = Tensor(
            output,
            requires_grad=self.requires_grad,
            _parents=(self,),
            _op="sigmoid",
        )

        def backward():
            if self.requires_grad:
                grad = result.grad * output * (1.0 - output)

                if self.grad is None:
                    self.grad = np.zeros_like(self.data)

                self.grad += grad

        result._backward = backward
        return result

    def gelu(self):
        # Constants for the tanh approximation of GELU
        alpha = np.sqrt(2.0 / np.pi)
        beta = 0.044715

        u = alpha * (self.data + beta * self.data ** 3)
        tanh_u = np.tanh(u)

        output = 0.5 * self.data * (1.0 + tanh_u)

        result = Tensor(
            output,
            requires_grad=self.requires_grad,
            _parents=(self,),
            _op="gelu",
        )

        def backward():
            if self.requires_grad:
                du_dx = alpha * (
                        1.0 + 3.0 * beta * self.data ** 2
                )

                derivative = (
                        0.5 * (1.0 + tanh_u)
                        + 0.5 * self.data
                        * (1.0 - tanh_u ** 2)
                        * du_dx
                )

                grad = result.grad * derivative

                if self.grad is None:
                    self.grad = np.zeros_like(self.data)

                self.grad += grad

        result._backward = backward
        return result






